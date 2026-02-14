
import boto3
import time

# Configuration
PROFILE_NAME = os.environ.get('AWS_PROFILE', 'default')
REGION_NAME = 'ap-south-1' # Defaulting to ap-south-1 as seen in previous files, but session usage usually picks it up from config if not specified.

def delete_all_endpoints():
    try:
        session = boto3.Session(profile_name=PROFILE_NAME, region_name=REGION_NAME)
        sm_client = session.client('sagemaker')
        
        print(f"Using profile: {PROFILE_NAME}")
        print(f"Region: {session.region_name}")
        
        # List all endpoints
        print("Listing all endpoints...")
        response = sm_client.list_endpoints(StatusEquals='InService')
        endpoints = response.get('Endpoints', [])
        
        # Also get endpoints that are creating or failing just in case
        response_all = sm_client.list_endpoints()
        endpoints_all = response_all.get('Endpoints', [])
        
        if not endpoints_all:
            print("No endpoints found.")
            return

        print(f"Found {len(endpoints_all)} endpoints.")
        
        for ep in endpoints_all:
            ep_name = ep['EndpointName']
            ep_status = ep['EndpointStatus']
            print(f"Deleting endpoint: {ep_name} (Status: {ep_status})")
            
            try:
                sm_client.delete_endpoint(EndpointName=ep_name)
                print(f" - Delete request sent for {ep_name}")
            except Exception as e:
                print(f" - Error deleting endpoint {ep_name}: {e}")
                
            # Optionally delete the endpoint configuration as well to keep it clean
            # We need to find the config name, usually it matches or we can describe the endpoint to find it
            # But if the endpoint is deleting, describe might fail or show it.
            # Typically endpoint config is separate. 
            # Trying to delete config with same name is a common guess, but let's just stick to endpoints as requested.
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    delete_all_endpoints()
