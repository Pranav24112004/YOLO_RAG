import urllib.parse
import os
import io
import fitz  # PyMuPDF
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions, ContentSettings
import config.settings as settings

def get_blob_service_client():
    return BlobServiceClient.from_connection_string(settings.AZURE_STORAGE_CONNECTION_STRING)

def get_source_sas_url(blob_name: str, page_number: int, expiry_minutes: int = 15) -> str:
    """
    Checks if a JPEG version of the requested page exists in the cache container.
    If not, downloads the source PDF, extracts the page as JPEG, and uploads it.
    Returns a short-lived SAS URL for the JPEG image.
    """
    conn_str = settings.AZURE_STORAGE_CONNECTION_STRING
    parts = dict(item.split('=', 1) for item in conn_str.split(';') if '=' in item)
    account_name = parts.get('AccountName')
    account_key = parts.get('AccountKey')
    
    source_container = settings.BLOB_CONTAINER
    cache_container = settings.PAGES_CONTAINER
    
    # Define the cached image name, e.g., "YOLOv1/p5.jpg"
    base_name = os.path.splitext(blob_name)[0]
    page_blob_name = f"{base_name}/p{page_number}.jpg"
    
    blob_service_client = get_blob_service_client()
    cache_container_client = blob_service_client.get_container_client(cache_container)
    page_blob_client = cache_container_client.get_blob_client(page_blob_name)

    # 1. Check if the page image already exists in the cache
    if not page_blob_client.exists():
        print(f"[Blob] Cache miss: Extracting page {page_number} from {blob_name}...")
        
        # 2. Download the original PDF
        source_container_client = blob_service_client.get_container_client(source_container)
        source_blob_client = source_container_client.get_blob_client(blob_name)
        
        pdf_bytes = source_blob_client.download_blob().readall()
        
        # 3. Extract the page as JPEG using PyMuPDF (fitz)
        # fitz is 0-indexed, so page_number - 1
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if (page_number - 1) < len(doc):
            page = doc.load_page(page_number - 1)
            # Use a matrix for better resolution (2.0 = 2x zoom/DPI)
            matrix = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
            
            # Save to an in-memory byte stream
            img_bytes_io = io.BytesIO(pix.tobytes("jpg"))
            
            # 4. Upload to the cache container
            page_blob_client.upload_blob(
                img_bytes_io, 
                overwrite=True,
                content_settings=ContentSettings(content_type="image/jpeg")
            )
        else:
            # Fallback or error if page is out of bounds
            print(f"[Blob] Error: Page {page_number} out of bounds for {blob_name}.")
            # For simplicity, we'll try to return the original PDF link if extraction fails
            # but ideally we should raise an error or handle it.
        doc.close()

    # 5. Generate SAS Token for the (newly created or existing) JPEG blob
    sas_token = generate_blob_sas(
        account_name=account_name,
        container_name=cache_container,
        blob_name=page_blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(minutes=expiry_minutes),
        content_type='image/jpeg',
        content_disposition='inline'
    )

    # 6. Construct Base URL
    safe_blob_name = urllib.parse.quote(page_blob_name)
    blob_url = f"https://{account_name}.blob.core.windows.net/{cache_container}/{safe_blob_name}"

    return f"{blob_url}?{sas_token}"


