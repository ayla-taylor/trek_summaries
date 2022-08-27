import pandas as pd
import requests
import wikipediaapi
from bs4 import BeautifulSoup

# wiki_html = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.HTML)


def df_change(df: pd.DataFrame) -> pd.DataFrame:
    """moves around the df so it is just the episode description"""
    df = pd.DataFrame(df).rename(columns={'Title': 'Summary'})
    df = df[df.Summary.apply(lambda x: len(str(x)) > 50)]
    return df[['Summary', 'Series']]


def scrape(url: str, seasons: int, series: str, voy=False, tng=False) -> pd.DataFrame:
    """gets the html and yields the episode descriptions as a dataframe"""
    # headers = {'Accept': 'text/html'}
    # page = wiki_html.page(url)
    # print(page.text)
    url_html = requests.get(url=url)
    soup = BeautifulSoup(url_html.text, 'html.parser')
    # if tng:
    #     print
    episodetable = soup.select('table', {'class': 'wikitable plainrowheaders wikiepisodetable'})
    # print(episodetable)
    dfs = pd.read_html(str(episodetable))
    if tng:
        dfs = [dfs[2]]
    else:
        dfs = dfs[1:seasons + 1] if seasons > 1 else [dfs[1]]
    if voy:
        dfs = dfs[:2] + dfs[4:]

    for df in dfs:
        df['Series'] = series
        yield df_change(df)


def main():
    filename = '../data/most_st_wikipages.txt'
    # this page has the columns in a different order than all the others, but it is the only one, so I just made an exception
    problem_voy = 'https://en.wikipedia.org/wiki/List_of_Star_Trek:_Voyager_episodes'
    problem_tng = 'https://en.wikipedia.org/wiki/Star_Trek:_The_Next_Generation_(season_7)'
    # scrape(url, 7, True)
    scraped = []
    with open(filename) as f:
        for line in f:
            season, url, series = line.strip().split()
            voy = True if url == problem_voy else False
            tng = True if url == problem_tng else False
            print("scraping:", url)
            scraped.append(pd.concat(list(scrape(url, int(season), series, voy, tng))))
    all_summaries = pd.concat(scraped)
    outfile = '../data/star_trek_episode_summaries_with_series.csv'
    print('Writing to', outfile, "...")
    all_summaries.to_csv(outfile, index=False)
    print('done')


if __name__ == '__main__':
    main()
