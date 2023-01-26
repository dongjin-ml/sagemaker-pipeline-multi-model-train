import os
import json
import argparse

class evalauator():
    
    def __init__(self, args):
                
        self.args = args
        self.strInOutPrefix = '/opt/ml/processing'
        self.strRegionName = self.args.region # boto3.Session().region_name
        
    def execution(self, ):
        
        print (f"Model URI: {self.args.s3_model_path}")
        print (f"Model Name: {self.args.model_name}")
        
        
        fAcc, fAuc, fPrec, fRec, fScore = 0.7, 0.1, 0.2, 0.9, 0.6
        report_dict = {
            "binary_classification_metrics": {
                "accuracy": {"value": fAcc, "standard_deviation": "NaN",},
                "auc": {"value": fAuc, "standard_deviation": "NaN"},
                "prec": {"value": fPrec, "standard_deviation": "NaN"},
                "rec": {"value": fRec, "standard_deviation": "NaN"},
                "fscore": {"value": fScore, "standard_deviation": "NaN"},
            },
        }

        print("Classification report:\n{}".format(report_dict))
        
        strOutputPath = "/opt/ml/processing/evaluation"
        
        evaluation_output_path = os.path.join(strOutputPath, "evaluation-" + self.args.model_name + ".json")
        print("Saving classification report to {}".format(evaluation_output_path))

        with open(evaluation_output_path, "w") as f:
            f.write(json.dumps(report_dict))
            
        print ("complete")


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3_model_path', type=str, default= "s3://")
    parser.add_argument("--region", type=str, default="ap-northeast-2")
    parser.add_argument("--model_name", type=str, default="model-default")
    
    args, _ = parser.parse_known_args()
    print("Received arguments {}".format(args))
    os.environ['AWS_DEFAULT_REGION'] = args.region
    
    
    evaluation = evalauator(args)
    evaluation.execution() 