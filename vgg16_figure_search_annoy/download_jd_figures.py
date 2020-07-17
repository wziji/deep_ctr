# encoding="utf-8"

from requests_html import HTMLSession
import re
import os
import time
import threading
import urllib.request


item_english_list = ["Mobile-phone", "T-shirt", "Milk", "Mask", "Headset", \
    "Wine", "Helmet", "Fan", "Sneaker", "Cup", \
    "Glasses", "Backpack", "UAV", "Sofa", "Bicycle", \
    "Cleanser", "Paper", "Bread", "Sausage", "Toilet", \
    "Book", "Tire", "Clock", "Mango", "Shrimp", \
    "Stroller", "Necklace", "Baby-bottle", "Yuba", "Pot"]


item_chinese_list = ["手机", "T恤", "牛奶", "口罩", "耳机", \
    "酒", "头盔", "风扇", "运动鞋", "杯子", \
    "眼镜", "背包", "无人机", "沙发", "自行车", \
    "洗面奶", "抽纸", "面包", "香肠", "马桶", \
    "书", "轮胎", "钟表", "芒果", "虾", \
    "童车", "项链", "奶瓶", "浴霸", "锅"]


session = HTMLSession()


def download_images(inx, key):

    for j in range(1, 10):
        
        time.sleep(2)
        url = 'https://search.jd.com/Search?keyword=%s&wq=%s&page=%s&s=90&click=0' % \
            (key, key, str(j))

        r = session.get(url)

        for i in range(1, 20):
            try:
                contain_pic_url = str(r.html.find('#J_goodsList > ul > li:nth-child('+str(i)+') > div > div > div.gl-i-tab-content > div.tab-content-item.tab-cnt-i-selected > div.p-img > a > img'))
                src_start = re.search('src',contain_pic_url).end() + 2
                src_end = int(re.search("'",contain_pic_url[src_start:]).start())
                pic_url = 'https:'+contain_pic_url[src_start:src_start + src_end]

                os.chdir('C:\\Users\\Desktop\\figures')
                pic = session.get(pic_url)
                open(item_english_list[inx]+'_page_'+str(j)+'_NO_'+str(i)+'.jpg','wb').write(pic.content)

            except:
                try:
                    contain_pic_url = str(r.html.find('#J_goodsList > ul > li:nth-child('+str(i)+') > div > div.p-img > a > img'))
                    src_start = re.search('src',contain_pic_url).end() + 2
                    src_end = int(re.search("'",contain_pic_url[src_start:]).start())
                    pic_url = 'https:'+contain_pic_url[src_start:src_start + src_end]

                    os.chdir('C:\\Users\\Desktop\\figures')
                    pic = session.get(pic_url)
                    open(item_english_list[inx]+'_page_'+str(j)+'_NO_'+str(i)+'.jpg','wb').write(pic.content)

                except:
                    pass

    print("Download %s done !!!" % item_english_list[inx])



def main():
    threads = []

    for inx, key in enumerate(item_chinese_list):
        thread = threading.Thread(target=download_images, args=(inx, key))
        threads.append(thread)
        thread.start()

    for i in threads:
        i.join()



start_time = time.time()
main()

print("use time is: ", time.time()-start_time)