
from io import StringIO, BytesIO
import requests
from lxml import etree

class Mineral:
  def __init__(self, url):

    # generate attribute dict
    self.scrape_text(url)
    self.image_urls = []

  def scrape_text(self, url):    
    self.attributes_dict = dict()
    page = requests.get(url)
    html = etree.HTML(page.content)
    tables = html.xpath('.//span[@id="ctl00_ContentPlaceHolder1_lblHardness"]')[0]
    print(tables)
    print(tables.attrib)

  def print_all_attributes(self):
    for att in self.attributes_dict:
      print(att + ":" + self.attributes_dict[att])

  def print_attribute(self, att):
    print(att + ":" + self.attributes_dict[att])

  def add_image(self, url):
    self.image_urls.append(url)

"""chemical_formula, composition, variable_formula, color, streak, hardness, crystal_system, crystal_forms, transparency, specific_gravity, luster, cleavage, fracture, tenacity, group, striking_features, evironment, rock_type, polularity, prevalence, demand """

#def add_image(self, url):

#def save_mineral(self, path='./'):

#Mineral('http://www.minerals.net/mineral/diopside.aspx')
 
