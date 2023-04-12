from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

"""
TODO
Find a way to get rid of the ad when doing this automatically
Ensure that the unit test works
"""

def ticker_to_website(ticker_list):
    website_strings = []
    for ticker in ticker_list:
        ticker_string = f"https://finance.yahoo.com/quote/{ticker}/history/"
        website_strings.append(ticker_string)
    return website_strings

def ticker_full_history(driver, ticker_string):
    elem = driver.find_element()

def grab_data():
    ticker_list = ['ETH-USD', 'BTC-USD', 'XMR-USD']
    ticker_string_list = ticker_to_website(ticker_list)

    driver = webdriver.Chrome()

    for ticker_string in ticker_string_list:
        driver.get(ticker_string)

        elem = driver.find_element(By.CLASS_NAME, "Va(m)! Mstart(8px) Stk($linkColor) Fill($linkColor) dateRangeBtn:h_Fill($linkActiveColor) dateRangeBtn:h_Stk($linkActiveColor) W(8px) H(8px) Cur(p)")

def unit_test():
    ticker = "ETH-USD"
    ticker_string = f"https://finance.yahoo.com/quote/{ticker}/history/"

    driver = webdriver.Chrome()
    driver.get(ticker_string)

    element = driver.find_element(By.CLASS_NAME, "Pos(r) D(ib) C($linkColor) Cur(p)")
    element.click()

    #max_history = driver.find_element(By.CLASS_NAME, "Py(5px) W(45px) Fz(s) C($tertiaryColor) Cur(p) Bd Bdc($seperatorColor) Bgc($lv4BgColor) Bdc($linkColor):h Bdrs(3px)")
    #max_history.click()

    #done_button = driver.find_element(By.CLASS_NAME, " Bgc($linkColor) Bdrs(3px) Px(20px) Miw(100px) Whs(nw) Fz(s) Fw(500) C(white) Bgc($linkActiveColor):h Bd(0) D(ib) Cur(p) Td(n)  Py(9px) Miw(80px)! Fl(start)")
    #done_button.click()

unit_test()

