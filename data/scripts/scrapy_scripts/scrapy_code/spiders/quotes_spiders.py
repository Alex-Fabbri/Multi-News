import scrapy
from twisted.internet import reactor
from scrapy.crawler import CrawlerProcess
from scrapy.crawler import CrawlerRunner
from scrapy.utils.log import configure_logging


class MySpider1(scrapy.Spider):
    name = "multi_news"

    def start_requests(self):
        # TODO change to the path to the inputs.txt or summaries.txt fils
        with open("inputs.txt", "r") as input:
            for line in input:
                line = line.split("\t")
                req = scrapy.Request(url=line[0].strip(), callback=self.parse)
                req.meta['filename'] = line[1].strip()
                yield req

    def parse(self, response):
        filename = response.meta['filename']
        # TODO change to the path where you want to store output html files
        with open(f"{filename}", "w") as f:
            f.write(response.text)
        self.log('Saved file %s' % filename)
