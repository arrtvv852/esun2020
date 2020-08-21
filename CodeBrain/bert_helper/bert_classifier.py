"""
Bert model for utterance sentiment analysis
"""
import os
import tensorflow as tf
from . import tokenization
from .run_classifier import BinaryProcessor
from .run_classifier import convert_examples_to_features
from .utils import name_filter
from .utils import get_sentences


class Model:
    """
    models to predict the sentiment of utterance using tf bert.
    """
    def __init__(self, model_dir="/app/models/news_classifier",
                 vocab_dir="/app/train_classifier/pretrain_bert_chinese",
                 tabu_dir="/app/data",
                 max_seq_length=510):
        self.max_seq_length = max_seq_length
        self.load_model(model_dir, vocab_dir, tabu_dir)

    def load_model(self, model_dir, vocab_dir, tabu_dir):
        """
        load required bert model
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=tf.Graph(), config=config)
        tf.saved_model.loader.load(self.sess, ['serve'], model_dir)
        self.processor = BinaryProcessor()
        self.label_list = self.processor.get_labels()
        self.vocab_file = os.path.join(vocab_dir, "vocab.txt")
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file,
                                                    do_lower_case=False)
        if tabu_dir is not None:
            tabu_file = os.path.join(tabu_dir, "rule_keywords.txt")
            with open(tabu_file, "r") as f:
                self.tabu_list = f.read().split("\n")
        else:
            self.tabu_list = []
    
    def name_check(self, name, content, threshold=0.25):
        """
        check if name suspecious inside the content
        """
        print(f"checking {name} ....")
        if not name_filter(name):
            return False
        sentences = get_sentences(content, name)
        if len(sentences) == 0:
            return False
        sentences = [f"{name}[SEP]{s}" for s in sentences]
        text = [[t] for t in sentences]
        predict_examples = self.processor._create_examples(text, "test")
        features = convert_examples_to_features(predict_examples,
                                                self.label_list,
                                                self.max_seq_length,
                                                self.tokenizer,
                                                "senti")
        batch_input_ids = [f.input_ids for f in features]
        batch_input_mask = [f.input_mask for f in features]
        batch_segment_ids = [f.segment_ids for f in features]
        pred = self.sess.run(['loss/out_probe:0'],
                             feed_dict={'input_ids_1:0': batch_input_ids,
                                        'input_mask_1:0': batch_input_mask,
                                        'segment_ids_1:0': batch_segment_ids})
        print(text)
        print(pred)
        if threshold > 0:
            score = pred[0].sum(0)
            if score[0]/(score[0] + score[1]) > threshold:
                print("True")
                return True
        else:
            score = pred[0].max(0)
            if score[0] > 0.95 or score[0] > score[1]:
                print("True")
                return True
        print("False")
        return False
    
    def inferance(self, text, adjust=0.3):
        """
        predict the sentiment
        """
        text = [[t] for t in text]
        predict_examples = self.processor._create_examples(text, "test")
        features = convert_examples_to_features(predict_examples,
                                                self.label_list,
                                                self.max_seq_length,
                                                self.tokenizer,
                                                "senti")
        batch_input_ids = [f.input_ids for f in features]
        batch_input_mask = [f.input_mask for f in features]
        batch_segment_ids = [f.segment_ids for f in features]
        pred = self.sess.run(['loss/out_probe:0'],
                             feed_dict={'input_ids_1:0': batch_input_ids,
                                        'input_mask_1:0': batch_input_mask,
                                        'segment_ids_1:0': batch_segment_ids})
        return self.rule_check(pred, text, adjust)
    
    def name_full_filter(self, name, content):
        """
        filter out the name with full content classification
        """
        text = f"{content[:(self.max_seq_length-10)]}[SEP]{name}"
        text = [[t] for t in text]
        predict_examples = self.processor._create_examples(text, "test")
        features = convert_examples_to_features(predict_examples,
                                                self.label_list,
                                                self.max_seq_length,
                                                self.tokenizer,
                                                "senti")
        batch_input_ids = [f.input_ids for f in features]
        batch_input_mask = [f.input_mask for f in features]
        batch_segment_ids = [f.segment_ids for f in features]
        pred = self.sess.run(['loss/out_probe:0'],
                             feed_dict={'input_ids_1:0': batch_input_ids,
                                        'input_mask_1:0': batch_input_mask,
                                        'segment_ids_1:0': batch_segment_ids})[0]
        if pred[0] > pred[1]:
            return True
        else:
            return False

    def rule_check(self, pred, texts, adjust):
        pred = pred[0]
        result = []
        for pre, text in zip(pred, texts):
            if pre[0] + adjust >= pre[1]:
                result.append(1)
            else:
                check = False
                for tabu in self.tabu_list:
                    if tabu in text:
                        check = True
                        break
                if check:
                    result.append(1)
                else:
                    result.append(0)
        return result

if __name__ == "__main__":
    model = Model("./models", ".", "./data", 510)
    text = ["36歲男子林致成將每瓶千元買入的「生物酵素-益菌飲品」，私自貼上「免疫細胞菌-養命活菌」標籤，加上他三寸不爛之舌，誆稱是專治癌症的神水，每瓶轉手賣3至5千元，王姓台商罹患口腔癌，被他說得放棄化療，買了17萬元，越喝身體越差，半年後多重器官衰竭死亡，台中地院依詐欺罪判處林8月徒刑，不法所得17萬元沒入。（示意圖）［記者楊政郡／台中報導］36歲林致成賺黑心錢！一瓶千元買入的「生物酵素-益菌飲品」，他私自貼上「免疫細胞菌-養命活菌」標籤，加上他三寸不爛之舌，誆稱是專治癌症的神水，每瓶賣3至5千元，其實僅是活菌、糖及黃豆粉等成份，王姓台商罹患口腔癌，被他說得放棄化療，花了17萬元買，越喝身體越差，半年後多重器官衰竭死亡，台中地院依詐欺罪判處林8月徒刑，不法所得17萬元沒入。判決書指出，林致成從2014年開始向陳富田，以1千元1瓶（960cc）買入「生物酵素-益菌飲品」，林私下委託印刷廠印製「免疫細胞菌-養命活菌」標籤，然後轉手每瓶賣3千到5千元。請繼續往下閱讀...2016年11月間林到西區大墩路羅姓友人住處推銷，正巧王姓台商寄住該處，林認為機不可失，憑三寸不爛之舌誆稱這種飲品專治癌症，有中國台商得癌症末期、瀕臨死亡都是喝他的飲料好的，還拿出從網路上搜尋而來好轉之反應資料給王觀看，還拿出他與人合照，聲稱是感謝他與他合照。王姓台商本身排斥化療，一聽到有專治癌症飲料，當場就花5千元買1瓶，回到廈門，又來電買30瓶，1瓶3000元，要他1天喝4次，1星期喝1瓶，而王來電反映有惡化，林一再宣稱這是「好轉反應」。林誆稱一般人要完全好，必須經過三次的「好轉反應」，就是變壞、變好；變壞再變好；變壞再變好，每次會有短暫舒適期，這是在等待下一次的排毒，變壞是正常現象，好像毒素要噴出來的感覺。王姓台商第二批再買25瓶，6萬5000元，這次是一直壞下去，沒有好起來了，2017年6月回林口長庚就醫，因多重器官衰竭死亡。林男坦承犯行，被告誇大療效，讓王姓台商懷抱希望，陷於錯誤而大量購買，王男惡化後還多次以排毒現像糖塞，詐騙手段明確，依詐欺罪判8月徒刑，不法所得17萬元沒入。不用抽 不用搶 現在用APP看新聞 保證天天中獎點我下載APP按我看活動辦法", "前總統馬英九被訴洩密案，一審判他無罪，二審逆轉判4月，受惠《刑事訴訟法》修法，馬英九可提一次上訴，後來最高法院撤銷有罪判決發回高院，高院更一審判決馬英九無罪確定。檢方起訴指控馬在2013年8月31日將時任檢察總長黃世銘向他報告，關於立法院長王金平司法關說案的偵查報告，洩漏給行政院長江宜樺、總統府副祕書長羅智強，9月4日再教唆黃將報告交付江宜樺，涉犯洩密罪等罪。台北地院認為馬英九的行為雖構成洩密罪，但屬於總統行使「院際調解權」，因阻卻違法判他無罪，高院認為此案與院際調解無關，批評馬英九未能恪遵法紀，逆轉將他改判4月、可易科罰金，但遭最高法院撤銷發回更審。上個月高院更一審進行最後辯論程序，馬英九堅稱無罪，他說檢察官對於犯罪的具體行為事實到底是什麼一直都說不清楚，不惜耗費司法資源就是要起訴他，他感到心痛；那些企圖以告他、起訴他來打擊他的人不會得逞，他對關說司法的人，沒有妥協的空間。檢方批評馬英九洩密還辯稱是為了公共利益，這樣的理由聽起來很華麗，但「公共利益」不是讓非法行 為合法化的大補帖；檢察官還舉電視劇「我們與惡的距離」，指責洩 密與司法關說一樣都是「惡」。對於洩密案宣判無罪確定，馬英九辦公室表示，對高等法院更一審的無罪判決感到欣慰，因為這項判決結果，不只攸關他個人的清白，更有助於確立憲法上總統應有的行政權限，讓國家領導人在任內能安心依憲依法治國。馬辦同時指出，本案審理兩年多來，檢察官如同最高法院發回判決所稱，對於馬前總統構成犯罪的具體行為事實為何，始終沒說清楚，明顯就是「先射箭再劃靶」，為達到起訴馬前總統的特定目的，不惜耗費司法資源。馬前總統雖因此遭受各種無謂糾纏，但一路走來捍衛司法獨立，無怨無悔，會繼續為杜絕司法關說奮戰，也再次呼籲蔡政府，不要再延宕「妨害司法公正罪」的制定，讓司法關說走入歷史，讓中華民國的司法能更受到人民的信任。(中時電子報)"]
    pred = model.inferance(text)