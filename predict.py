import os 
import json
import configparser
import warnings
from train import trainer 
from bert_model import get_model
from data_loader import NER_Dataset
from torch.utils.data import DataLoader
from utils import create_logger,ConfigLogger
import logging

warnings.filterwarnings('ignore')

config = configparser.ConfigParser() 
config.optionxform = str
config.read("config.conf")
if __name__ == "__main__":
    create_logger(config['model']['mode']).logger_init()
    logger = logging.getLogger(__name__)

    logger.info('===== import configuration =====')
    config_logger = ConfigLogger(logging)
    config_logger(config)

    model_folder_path = config['pytorch_model']['pytorch_Bert_model']
    model_config_path = os.path.join(model_folder_path,'config.json')
    model_path = os.path.join(model_folder_path,'pytorch_model.bin')

    with open(model_config_path, newline='') as jsonfile:
        model_config = json.load(jsonfile)

    # import data
    batch_size = int(config['hyperparameter']['batch_size'])
    sentence = ['巴 基 斯 坦 對 ５ 國 聯 合 公 報 表 示 歡 迎 ， 但 拒 絕 了 １ １ ７ ２ 號 決 議 ； 印 度 則 對 上 述 兩 個 文 件 予 以 拒 絕 。',
                '六 十 年 代 初 ， 吳 先 生 到 了 北 京 ， 還 到 我 家 做 客 。',
                '在 克 林 頓 總 統 的 家 鄉 （ 附 圖 片 １ 張 ） 他 最 後 說 ： “ 美 國 人 民 希 望 中 國 人 民 安 康 ， 我 們 想 更 多 地 了 解 中 國 的 今 天 和 昨 天 ， 也 希 望 更 多 地 參 與 中 國 的 明 天 。',
                '疫 情 前 後 數 據 對 比 報 告：南 韓 大 型 企 業 共 裁 員 萬 人',
                '香 港 特 區 終 審 法 院 及 終 審 法 院 首 席 法 官 香 港 特 別 行 政 區 臨 時 立 法 會 今 天 舉 行 的 第 八 次 全 體 會 議 ， 一 致 同 意 任 命 李 國 能 為 香 港 特 區 終 審 法 院 首 席 法 官 。',
                '畢 加 索 於 １ ８ ８ １ 年 出 生 在 西 班 牙 南 部 馬 拉 加 ， 父 親 是 位 畫 師 ， 他 以 自 己 的 直 覺 意 識 到 兒 子 的 藝 術 天 賦 ， 便 鼓 勵 畢 加 索 去 巴 塞 羅 那 和 馬 德 裡 學 習 美 術 。',
                '同 學 會 協 助 王 橞 瀴 進 行 說 明 ， 你 會 嗎 ? 王 橞 瀴 會 協 助 您',
                '江 主 席 是 應 美 國 總 統 克 林 頓 的 邀 請 ， 從 １ ０ 月 ２ ６ 日 起 對 美 國 進 行 國 事 訪 問 的 。',
                ' 老 牌 服 飾 業 者 涉 詐 貸 1 6 家 銀 行 踩 雷 12.25 億 台 銀 是 最 大 苦 主',
                '解 讀 台 積 電 張 忠 謀 在 演 講 的 4 大 關 鍵 ！ 一 場 劉 德 音 、 魏 哲 家 都 沒 錯 過 的 談 話']
    test_dataset = NER_Dataset(config).load_data('test',input_sentences = sentence)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False,#num_workers = 8,pin_memory = True,
                                 collate_fn=test_dataset.collate_fn)
    
    # train_dataset = NER_Dataset(config).load_data('train')
    # train_loader = DataLoader(train_dataset, batch_size = batch_size,
    #                          shuffle=True,#num_workers = 8,pin_memory = True,
    #                          collate_fn=train_dataset.collate_fn)
    # 
    # val_dataset = NER_Dataset(config).load_data('val')
    # val_loader = DataLoader(val_dataset, batch_size=batch_size,
    #                          shuffle=True,#num_workers = 8,pin_memory = True,
    #                          collate_fn=val_dataset.collate_fn)

    # import model
    selected_model = model_config['architectures'][0]
    config['model']['selected_model'] = selected_model
    model = get_model(config = config)
    

    result = trainer(model,config,logger)
    
    # result.import_data((train_loader,val_loader),(len(train_dataset),len(val_dataset)))
    # result.selected_model = selected_model
    # metrics = result.evaluate()

    result.import_data(test_loader,len(test_dataset))
    AA = result.predict()