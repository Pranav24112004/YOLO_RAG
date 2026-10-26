
from azure.cosmos import CosmosClient
import config.settings as settings


def get_cosmos_client():
    return CosmosClient(settings.AZURE_COSMOS_ENDPOINT, settings.AZURE_COSMOS_KEY)


def get_chat_container():
    client = get_cosmos_client()
    database = client.get_database_client(settings.DATABASE_NAME)
    return database.get_container_client(settings.CONTAINER_NAME)


from azure.cosmos.exceptions import CosmosResourceNotFoundError

def get_session_history(session_id):
    """Fetch the message array for a session document where id=session_id."""
    container = get_chat_container()
    try:
        item = container.read_item(item=session_id, partition_key=session_id)
        print("item: ", item)
        return item.get("messages", [])
    except CosmosResourceNotFoundError:
        return []
        
def list_sessions(limit=20):
    """Retrieve session IDs and timestamps for sidebar list."""
    container = get_chat_container()
    
    # Query for id and using _ts for more reliable sorting (internal timestamp)
    query = "SELECT c.id, c.last_updated, c._ts FROM c ORDER BY c._ts DESC OFFSET 0 LIMIT @limit"
    parameters = [{"name": "@limit", "value": limit}]
    
    items = list(container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True
    ))
    return items
