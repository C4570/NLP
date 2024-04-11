# !pip install pandas
# !pip install beautifulsoup4
# !pip install selenium


# Librerías
import pandas as pd

import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait


# Código
with webdriver.Firefox() as driver:

    # Abrir la página de El Ministerio de Turismo y Deportes de la Nación
    driver.get("https://tableros.yvera.tur.ar/")
    wait = WebDriverWait(driver, 10)
    
    # Clickeaamos en la imagen que dice "Indicadores ODS (Objetivos de Desarrollo Sostenible)"
    driver.find_element(By.XPATH, "/html/body/d-article/div[2]/div/div/div[17]/a/img").click()

    # Obtenemos la URL de la página a la cual fuimos redirigidos
    current_url = driver.current_url
    
    
    # Extraigo los valores de los tableros y los guardo en una tabla
    response = requests.get(current_url, verify=False)
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    if response.status_code == 200:
        html = response.text
        summary_div = soup.find('div', {'id': 'shiny-tab-summary'})

        data = {
            'Texto': [],
            'Numero': []
        }

    if summary_div:
        relevant_divs = summary_div.find_all('div', {'class': 'col-sm-4'})
        div = relevant_divs[0]
        cubos = []
        for div in relevant_divs:
          p_elements = div.find_all('p')

          if (len(p_elements)) > 3:
            cubos.append(p_elements[:2])
            cubos.append(p_elements[3:5])

          else:
            p_element = p_elements[:2]
            cubos.append(p_element)

        for cubo in cubos:
          texto = cubo[1].get_text(strip=True)
          data['Texto'].append(texto)
          numero = cubo[0].get_text(strip=True)
          data['Numero'].append(numero)

    df = pd.DataFrame(data)
    
    print(df)
    
    
    # Clickeamos en el botón "Metodología"
    driver.find_element(By.XPATH, '//*[@id="sidebarItemExpanded"]/ul/li[3]/a').click()


    # Extraigo la metodología y la imprimo
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')

    box_body_divs = soup.find_all('div', {'class': 'box-body'})

    if box_body_divs:
      div = box_body_divs[-1]
      texto = ""
      for child in div.children:
        if child.name == 'p':
          texto += child.get_text(strip=True) + "\n"
        elif child.name == 'b':
          texto += child.get_text(strip=True) + "\n"
        elif child.name == 'ul':
          for p in child.find_all('p'):
            for b in ul_child.find_all('b'):
              texto += b.get_text(strip=True) + "\n"

      print(texto)