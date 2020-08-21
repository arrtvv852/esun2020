
import scrapy


class NewsCrawlerItem(scrapy.Item):
    url = scrapy.Field()
    published_time = scrapy.Field()
    title = scrapy.Field()
    content = scrapy.Field()
    keywords = scrapy.Field()
    category = scrapy.Field()
    item_type = scrapy.Field()
    pass


class Spider(scrapy.Spider):
    name = 'chinatimes'
    allowed_domains = ['www.chinatimes.com']
    check_url = True
    proxy = False
    kafka_topics = ['news']
    item_type = 'news'
    total_pages = 20

    def start_requests(self):
        category_urls = ('https://www.chinatimes.com/realtimenews/?chdtv')
        for url in category_urls:
            for page in range(1, self.total_pages):
                yield scrapy.Request(url='{u}&page={p}'.format(u=url, p=page))

    def parse(self, response):
        for url in response.xpath(
                '//ul[contains(@class,"list-style-none")]/li[not(@id)]//h3/a/@href').extract():
            news_url = 'https://www.chinatimes.com{}?chdtv'.format(url)
            yield scrapy.Request(news_url, callback=self.get_news)

    def get_news(self, response):
        news_item = NewsCrawlerItem({'url': response.url,
                                     'item_type': self.item_type})
        news_item['published_time'] = response.xpath(
            '//meta[@property="article:published_time"]/@content').get()
        news_item['title'] = response.xpath(
            '//meta[@property="og:title"]/@content').get()
        news_item['content'] = response.xpath(
            '//div[@class="article-body"]/p//text()').getall()
        news_item['category'] = response.xpath(
            '//meta[@name="section"]/@content').get()
        news_item['keywords'] = response.xpath(
            '//div[@class="article-hash-tag"]/span[@class="hash-tag"]/a/text()').getall()
        return news_item
