import re

class PttExtractor:

    def extract(self, content):
        def post_proc(label):
            label = re.sub('^( |　)*', '', label)
            label = re.sub('( |　)*$', '', label)
            label = re.sub('\t', ' ', label)
            label = re.sub(' {2,}', ' ', label)
            return label


        weekdays="(Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
        months="(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        dates="[0-9]{1,2} [0-9]{2}:[0-9]{2}:[0-9]{2} [0-9]{4}"
        sep="[- 　\t\.]"
        ename="(\n|消費|時"+sep+"*間|地"+sep+"*址|營業|價位|每人|平均|電"+sep+"*話|【)"
        eaddr="(\n|消費|時"+sep+"*間|餐廳|店名|名稱|營業|價位|每人|平均|電"+sep+"*話|交通|【)"
        ephone="(\n|消費|時"+sep+"*間|餐廳|店名|名稱|營業|價位|每人|平均|地"+sep+"*址|【)"
        lq="[([（【]"
        rq="[])）】]"
        digit="[0-9０１２３４５６７８９]"
        ip="[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}"
        url="(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/[^ \n]*)?"


        post_date = ""
        name = ""
        visit_date = ""
        address = ""
        phone1 = ""
        phone2 = ""
        
        #print(content)
        content = re.sub(ip, "", content) #remove ip as phone.
        content = re.sub(url, "", content) #remove url as phone.
        #print(content)
        
        # 發文時間
        # 時間Thu Jul 11 06:16:29 2019
        pat = '時間('+weekdays+" "+months+" "+dates+')'
        obj=re.search(pat, content)
        if obj:
            post_date = obj.group(1)
            post_date = post_proc(post_date)
            
        # 餐廳名稱
        # 餐廳名稱：蛋黃哥不想上班咖啡廳
        pat = lq+'*(餐廳名稱|店'+sep+'*名|飯店名稱)'+rq+'*'+'(：|:)'+sep+'*(.*?)'+sep+'*'+ename
        obj=re.search(pat, content)
        if obj:
            name = obj.group(3)
            name = post_proc(name)
        else:
            #標題[廣宣] 台北信義吳興商圈平價丼飯肉叔叔燒肉店時間
            obj = re.search('標題(.*?)時間', content)
            if obj:
                name = obj.group(1)
                name = re.sub('\[.*\]', '', name)
                name = re.sub('Fw(:|：)*', '', name)
                name = re.sub('轉(寄|貼|帖)(:|：)', '', name)
                name = post_proc(name)

        
        # 地址
        # 地址：新竹市北區湳雅街91-2號(大魯閣湳雅廣場一樓)
        pat = lq+'*地'+sep+'*址'+rq+'*(：|:)'+sep+'*(.*?)'+sep+'*'+eaddr
        obj=re.search(pat, content)
        if obj:
            address = obj.group(2)
            address = post_proc(address)
    
        # 電話
        pat = lq+'*電'+sep+'*話'+rq+'*(：|:)'+sep+'*(.*?)'+sep+'*'+ephone
        obj=re.search(pat, content)
        if obj:
            phone1 = obj.group(2)
        pat = lq+'?'+sep+'*'+digit+"{2,4}"+sep+'*'+rq+'?'+sep+'*'+digit+'{3,4}' \
        +sep+'*'+digit+'{3,4}'
        obj=re.search('('+pat+')', content)
        if obj:
            #print(obj.group(1))
            phone2 = obj.group(1)
            phone2 = post_proc(phone2)

        if phone1 != "":
            phone = phone1
        else:
            phone = phone2
        if phone in ["無", "NA", "N/A", "na", "n/a", "不知", "沒有"]:
            phone = ""
            
        # 消費時間
        pat = lq+'*(消費|用餐|試吃|拜訪|到店|食用|旅遊|體驗)(時間|日期|消費)'+rq+'* *(：|:|→)?'+sep+'*([0-9０１２３４５６７８９年/月/日／]+)'
        obj=re.search(pat, content)
        if obj:
            visit_date=obj.group(4)
            visit_date=post_proc(visit_date)
        if visit_date == "":
            visit_date=post_date
        
        #print("發文時間: "+post_date)
        #print("餐廳名稱: "+name)
        #print("地址: "+address)
        #print("電話: {}/{}".format(phone1, phone2))
        #print("消費時間: "+visit_date)
        #print("")

        return name, phone, address, visit_date
