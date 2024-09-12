
pass # !find /content -maxdepth 1 -type d \( -name training_logs -o -name nas_results -o -name plots -o -name model_metrics -o -name tasks \) | zip -r results.zip -@

