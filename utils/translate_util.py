import os
import json
from google.cloud import translate
import pandas as pd

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/ilaif/credentials/ITC-BI-workshop.json'

t_client = translate.Client()


def translate(input_path, output_path, target_lang):
    texts = json.load(open(input_path, 'r'))
    res = {t: t_client.translate(t, target_language=target_lang)['translatedText'] for t in texts if not pd.isnull(t)}
    json.dump(res, open(output_path, 'w'))


if __name__ == '__main__':
    names = ['parent_category_name', 'category_name', 'param_1', 'param_2', 'param_3']

    for name in names:
        translate(name + '.json', name + '_en.json', 'en')
