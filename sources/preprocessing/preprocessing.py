import os
import sys
import boto3
import argparse
import subprocess
import pandas as pd
from sklearn.model_selection import train_test_split

def install_packages():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "/opt/ml/processing/input/requirements/requirements.txt",])
    
class preprocess():
    
    def __init__(self, args):
        
        from spacy.lang.en import English
        
        self.args = args
        self.strInOutPrefix = '/opt/ml/processing'
        nlp = English()
        self.tokenizer = nlp.tokenizer
        self.index_to_label = {0: 'NotHelpful', 1: 'Helpful'} 
      
    def _labelize_df(self, df):
        
        return '__label__' + df['is_helpful'].apply(lambda is_helpful: self.index_to_label[is_helpful])

    def _tokenize_sent(self, sent, max_length=1000):
        
        return ' '.join([token.text for token in self.tokenizer(sent)])[:max_length]

    def _tokenize_df(self, df):

        return (df['review_headline'].apply(self._tokenize_sent) + ' ' + 
                df['review_body'].apply(self._tokenize_sent))
    
    def execution(self, ):
        
        input_data_path = os.path.join(self.strInOutPrefix, "input", args.input_name)
        print (f'isfile: {os.path.isfile(input_data_path)}')
        
        if not os.path.isdir(os.path.join(self.strInOutPrefix, 'output', 'train')): os.makedirs(os.path.join(self.strInOutPrefix, 'output', 'train'))
        if not os.path.isdir(os.path.join(self.strInOutPrefix, 'output', 'validation')): os.makedirs(os.path.join(self.strInOutPrefix, 'output', 'validation'))
        if not os.path.isdir(os.path.join(self.strInOutPrefix, 'output', 'test')): os.makedirs(os.path.join(self.strInOutPrefix, 'output', 'test'))
        print (f'input_data_path: {input_data_path}')
        

        df_reviews = pd.read_csv(input_data_path, compression='gzip', error_bad_lines=False, sep='\t', \
                                 usecols=['product_id', 'product_title', \
                                          'review_headline', 'review_body', 'star_rating', \
                                          'helpful_votes', 'total_votes']).dropna()

        df_reviews = df_reviews[df_reviews['total_votes'] >= 5]
        df_reviews['helpful_score'] = df_reviews['helpful_votes'] / df_reviews['total_votes']
        df_reviews['sentiment'] = pd.cut(df_reviews['star_rating'], bins=[0,2,3,6], labels=['Negative','Nuetral','Positive'])
        df_reviews.describe()

        df_votes =  df_reviews.groupby('product_id').agg({'product_id': 'count', 'helpful_votes': 'sum', 'total_votes': 'sum'})
        df_votes.describe()

        min_reviews = 1
        min_helpful = 5
        df_votes = df_votes[(df_votes['product_id']>=min_reviews) & (df_votes['helpful_votes']>=min_helpful)]

        df_reviews = df_reviews.merge(df_votes, how='inner', left_on='product_id', right_index=True, suffixes=('','_total'))
        df_reviews.info()

        df_reviews['is_helpful'] = (df_reviews['helpful_score'] > 0.80)
        df_reviews['is_helpful'].sum()/df_reviews['is_helpful'].count()

        train_df, val_df = train_test_split(df_reviews, test_size=0.1, random_state=42) 
        val_df, test_df = train_test_split(val_df, test_size=0.5, random_state=42)
        print('split train: {}, val: {}, test: {} '.format(train_df.shape[0], val_df.shape[0], test_df.shape[0]))

        test_df.to_csv(os.path.join(self.strInOutPrefix, 'output', 'test', 'test.csv'), index=False, header=True)
        
        train_text = self._labelize_df(train_df) + ' ' + self._tokenize_df(train_df)
        val_text = self._labelize_df(val_df) + ' ' + self._tokenize_df(val_df)
        
        train_text.to_csv(os.path.join(self.strInOutPrefix, 'output', 'train', 'train.txt'), index=False, header=False)
        val_text.to_csv(os.path.join(self.strInOutPrefix, 'output', 'validation', 'validation.txt'), index=False, header=False)
        
        print (val_text)
        
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_name", type=str, default="reviews.tsv.gz")
    parser.add_argument("--region", type=str, default="ap-northeast-2")
    args, _ = parser.parse_known_args()
           
    print("Received arguments {}".format(args))
    os.environ['AWS_DEFAULT_REGION'] = args.region
    
    install_packages()
    
    prep = preprocess(args)
    prep.execution()