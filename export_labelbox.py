import labelbox
from labelbox import Client, Project
import json
from datetime import datetime
import os

# Initialize the Labelbox client
client = Client(api_key=os.environ.get("RUNPOD_SECRET_LABELBOX_API_KEY"))

# Get your project by ID
# Replace with your actual project ID
PROJECT_ID = os.environ.get("LABELBOX_PROJECT_ID", "cmehk2qe503kq07zl3imoccwj")

try:
    # Fetch the project
    project = client.get_project(PROJECT_ID)
    print(f"Connected to project: {project.name}")
    print(f"Project ID: {project.uid}")
    print("-" * 50)
    
    # Set the export params to include/exclude certain fields
    export_params = {
        "attachments": True,
        "metadata_fields": True,
        "data_row_details": True,
        "project_details": True,
        "label_details": True,
        "performance_details": True,
        "interpolated_frames": True
    }
    
    # Note: Filters follow AND logic, so typically using one filter is sufficient
    # You can adjust these dates or comment out the filters entirely
    filters = {
        "last_activity_at": ["2000-01-01 00:00:00", "2050-01-01 00:00:00"],
        # "workflow_status": "Done"  # Uncomment and set to your workflow status if needed
    }
    
    print("Starting export with parameters:")
    print(json.dumps(export_params, indent=2))
    print("\nFilters:")
    print(json.dumps(filters, indent=2))
    print("-" * 50)
    
    # Create export task
    export_task = project.export(params=export_params, filters=filters)
    
    # Wait for export to complete
    print("Waiting for export to complete...")
    export_task.wait_till_done()
    print("Export completed!")
    print("-" * 50)
    
    # Method 1: Stream the export using a callback function
    print("\n=== METHOD 1: Streaming with callback ===")
    counter = {"count": 0}  # Using dict to allow modification in nested function
    
    def json_stream_handler(output: labelbox.BufferedJsonConverterOutput):
        counter["count"] += 1
        print(f"\n--- Data Row {counter['count']} ---")
        # Pretty print the JSON
        print(json.dumps(output.json, indent=2))
        # Uncomment the line below if you want to see just the first few items
        # if counter["count"] >= 3:
        #     return False  # Stop streaming after 3 items
    
    export_task.get_buffered_stream(stream_type=labelbox.StreamType.RESULT).start(
        stream_handler=json_stream_handler
    )
    
    # Method 2: Collect all exported data into a list
    print("\n\n=== METHOD 2: Collecting all data ===")
    export_json = [data_row.json for data_row in export_task.get_buffered_stream()]
    
    print(f"\nTotal number of exported data rows: {len(export_json)}")
    
    # Print first few items if any exist
    if export_json:
        print("\nFirst exported item (detailed view):")
        print(json.dumps(export_json[0], indent=2))
        
        # Print summary of all items
        print(f"\n=== Summary of {len(export_json)} exported items ===")
        for idx, item in enumerate(export_json[:5], 1):  # Show first 5 items
            print(f"\nItem {idx}:")
            print(f"  - Data Row ID: {item.get('data_row', {}).get('id', 'N/A')}")
            print(f"  - External ID: {item.get('data_row', {}).get('external_id', 'N/A')}")
            print(f"  - Media Type: {item.get('media_attributes', {}).get('media_type', 'N/A')}")
            
            # Check for labels
            projects_info = item.get('projects', {})
            if projects_info:
                for proj_id, proj_data in projects_info.items():
                    labels = proj_data.get('labels', [])
                    print(f"  - Number of labels: {len(labels)}")
                    if labels:
                        for label in labels[:2]:  # Show first 2 labels
                            print(f"    • Label ID: {label.get('id', 'N/A')}")
                            annotations = label.get('annotations', {}).get('frames', [])
                            print(f"      Frames with annotations: {len(annotations)}")
        
        if len(export_json) > 5:
            print(f"\n... and {len(export_json) - 5} more items")
    else:
        print("\nNo data was exported. This could mean:")
        print("  - The project has no labeled data")
        print("  - The filters are too restrictive")
        print("  - The project has no data rows")
    
    # Print export statistics
    print("\n=== Export Statistics ===")
    print(f"File size: {export_task.get_total_file_size(stream_type=labelbox.StreamType.RESULT)} bytes")
    print(f"Line count: {export_task.get_total_lines(stream_type=labelbox.StreamType.RESULT)}")
    
    # Optional: Save to file
    save_to_file = True  # Set to True if you want to save the export
    if save_to_file and export_json:
        # Save as NDJSON (newline-delimited JSON)
        filename = f"labelbox_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ndjson"
        with open(filename, 'w') as f:
            for item in export_json:
                # Write each item as a single line of JSON
                f.write(json.dumps(item) + '\n')
        print(f"\nExport saved to: {filename}")
        print(f"Format: NDJSON (Newline-Delimited JSON)")
        print(f"Total lines: {len(export_json)}")
        
        # Also save a pretty-printed version for easier reading (optional)
        save_pretty_version = False  # Set to True if you also want a formatted version
        if save_pretty_version:
            pretty_filename = f"labelbox_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}_pretty.json"
            with open(pretty_filename, 'w') as f:
                json.dump(export_json, f, indent=2)
            print(f"Pretty-printed version saved to: {pretty_filename}")

except Exception as e:
    print(f"Error occurred: {str(e)}")
    print("\nTroubleshooting tips:")
    print("1. Make sure your API key is set correctly")
    print("2. Verify the PROJECT_ID is correct")
    print("3. Check that you have access to the project")
    print("4. Ensure the project has video data with annotations")