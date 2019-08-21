import collections

import submission as submission
import pickle
import re

# input_dir = './BBC_Data.zip'
# data_file = submission.process_data(input_dir)

data_file='data_file'
with open(data_file, 'rb') as f:
    data_list = pickle.load(f)

print(' '.join(data_list[:100]))

# for ent in doc.ents:
#     new_article = re.sub(r'\b{}\b'.format(ent.text), ent.label_, new_article)
# for token in doc:
#     if token.text!=token.lemma_:
#         new_article = re.sub(r'\b{}\b'.format(token.text), token.lemma_, new_article)




