
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

import config.settings as settings

def get_search_client():
    return SearchClient(
        endpoint=settings.AZURE_SEARCH_ENDPOINT,
        index_name=settings.INDEX_NAME,
        credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY)
    )

def get_index_client():
    return SearchIndexClient(
        endpoint=settings.AZURE_SEARCH_ENDPOINT,
        credential=AzureKeyCredential(settings.AZURE_SEARCH_KEY)
    )
