
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import config.settings as settings

def get_doc_intel_client():
    return DocumentAnalysisClient(
        endpoint=settings.AZURE_DOC_INTEL_ENDPOINT,
        credential=AzureKeyCredential(settings.AZURE_DOC_INTEL_KEY)
    )
