import sys
import datetime
import time
import json
sys.path.append("/app/ner")
sys.path.append("/app/bert_helper")
sys.path.append("/app/ner/src")
import Utils
from utils import content_split
import ner_extractor as nerExtractor
from bert_helper.bert_classifier import Model


NEWS_CLASSIFIER = Model(model_dir="/app/models/news_classifier", max_seq_length=511)
NAME_CLASSIFIER = Model(model_dir="/app/models/names_classifier", max_seq_length=50)
# NAME_FULL_CLASSIFIER = Model(model_dir="/app/models/names_full_classifier", max_seq_length=511)

def test():
    article = "台北地檢署15日依違反證交法罪嫌起訴金管會前副主委呂東英之子呂建安圖為北檢外觀記者林裕豐攝記者劉昌松台北報導金管會前副主委呂東英之子呂建安涉在2013年間利用仲介股權買賣業務之便先買後賣上櫃公司佳營電子股票獲利22萬元15日被台北地檢署依證券交易法內線交易罪嫌起訴但考量呂事後坦承犯行並交出不法所得請求法院減輕其刑檢方起訴指出2012年間友尚公司希望精簡處分持有的佳營已下櫃股權委託境外公司尋找買家當時東南亞投資顧問公司副總經理呂建安得知後居間仲介找到有買家願意以每股165元價格收購當時市價約12元的佳營呂建安因這項仲介業務得知內線後竟在2013年1月到5月間陸續買進再賣出佳營股票直到這項收購資訊公布佳營的股權有4412一口氣轉到英屬維京群島天悅公司名下呂建安靠內線不法獲利22萬元在此同時友尚母公司大聯大的財務副總黃淑頻則是在重大訊息公佈後18小時內違反證交法規定賣股獲利6800元檢調追查後呂黃都坦誠犯行並繳回犯罪所得15日依違反證交法將2人起訴此外全案一度傳出呂利用這次股權交易向佳營電子負責人吳成友尚公司負責人曾國輝收取逾850萬元回扣疑有特別背信罪責部分經檢調追查其實是友尚公司要透過呂的東南亞投顧交付給別人的仲介費錢沒有遭私吞這部分處分不起訴"
    result = nerExtractor.extract(article)
    print(result)
    
def log_article(msg):
    file_name = "/app/log/log_{}.txt".format(datetime.datetime.now().strftime('%Y-%m-%d'))
    formatted_msg = "{}\t{}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg)
    Utils.file_append(file_name, formatted_msg + "\n")
    
def log_result(article, has_aml, names):
    file_name = "/app/log/result_{}.txt".format(datetime.datetime.now().strftime('%Y-%m-%d'))
    formatted_msg = "{}\t{}\t{}\t{}\t{}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), len(article), article, has_aml, names)
    Utils.file_append(file_name, formatted_msg + "\n")

def do_single_extract(article):
        
    if NEWS_CLASSIFIER.inferance([article])[0]:
        
        names = nerExtractor.extract(article)
        predicted_names = []
        for name in names:
             if NAME_CLASSIFIER.name_check(name, article):
#              if True:
                predicted_names.append(name)
        return {'article': article,
                       'has_aml': True,
                       'ner_names': names,
                       'predicted_names': predicted_names,
                      }
    else:
        return {'article': article,
                       'has_aml': False,
                       'ner_names': [],
                       'predicted_names': [],
                      }


def extract(article, do_split=True):
    
    timestamp = int(time.time()*1000)
    if do_split:
        splited_articles = content_split(article, 511)
    else:
        splited_articles = [article]
    
    ner_names = []
    names = []
    results = []
    has_aml = False
    
    for splited_article in splited_articles:
        result = do_single_extract(splited_article)
        results.append(result)

        if result:
            if result["has_aml"] == True:
                has_aml = True

            for name in result["ner_names"]:
                ner_names.append(name)
            
            for name in result['predicted_names']:
                names.append(name)
    
    names = Utils.unique(names)
    ner_names = Utils.unique(ner_names)
    
    if len(names) == 0 and has_aml == True:
        names = ner_names
    
    result = {'timestamp': timestamp,
               'has_aml': has_aml,
               'ner_names': ner_names,
               'names': names,
               'split_cnt': len(splited_articles),
               'results': results,
               'article': article
             }

    if has_aml:
        log_result(article, 1, names)
    else:
        log_result(article, 0, names)

    log_article(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return names
