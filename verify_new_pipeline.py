from neuroswift.data_pipeline import UniversalSchemaMapper, TrainPair
import json

def test_pipeline():
    # 1. Test Substring PII Filter (should catch 'message_id')
    pii_obj = {"instruction": "Hello", "response": "Hi", "message_id": "12345"}
    pair = UniversalSchemaMapper.map_obj(pii_obj, "test_source")
    print(f"PII Object 'message_id' Result (should be None): {pair}")
    
    # 2. Test UUID Avoidance
    uuid_val = "a41f23d1-dc75-497d-8d99-a5b7b061022f"
    bad_obj = {"instruction": uuid_val, "response": "Real answer"}
    pair = UniversalSchemaMapper.map_obj(bad_obj, "test_source")
    print(f"UUID Value Object Result (should reject or pick better keys): {pair}")
    
    # 3. Test Valid Schema
    valid_obj = {"instruction": "What is NeuroSwift?", "response": "A framework."}
    pair = UniversalSchemaMapper.map_obj(valid_obj, "test_source")
    print(f"Valid Object Result: {pair}")

if __name__ == "__main__":
    test_pipeline()
