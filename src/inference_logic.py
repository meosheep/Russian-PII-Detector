import pandas as pd
import torch
from transformers import pipeline

def run_inference(model_path, test_data_path, output_csv):
    """Loads a trained model and runs inference on the test dataset."""
    # Initialize the NER pipeline with simple aggregation to merge subword tokens
    nlp_ner = pipeline(
        "ner", 
        model=model_path, 
        tokenizer=model_path, 
        aggregation_strategy="simple", 
        device=0 if torch.cuda.is_available() else -1
    )

    test_df = pd.read_csv(test_data_path)
    
    def get_predictions(text):
        """Processes a single text string and returns formatted entity tuples."""
        if not isinstance(text, str) or len(text.strip()) == 0:
            return []
        results = nlp_ner(text)
        # Format: (start_index, end_index, entity_type)
        return [(int(res['start']), int(res['end']), res['entity_group']) for res in results]

    print("Running inference...")
    test_df['Prediction'] = test_df['text'].apply(get_predictions)
    
    # Prepare final submission file
    submission = test_df[['id', 'Prediction']] if 'id' in test_df.columns else test_df[['Prediction']]
    submission.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")