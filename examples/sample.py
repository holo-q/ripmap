"""Sample Python file for testing tag extraction."""

# Top-level constant
MAX_RETRIES = 5

class DatabaseConnection:
    """Manages database connections with retry logic."""
    
    def __init__(self, host, port):
        self.host = host
        self.port = port
    
    def connect(self):
        """Establish connection to the database."""
        pass
    
    def disconnect(self):
        """Close the database connection."""
        pass

def calculate_total(items):
    """Calculate the total of all items."""
    return sum(items)

def process_batch(batch_size=100):
    """Process items in batches."""
    pass
