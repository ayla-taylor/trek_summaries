import pandas as pd
import requests
from bs4 import BeautifulSoup


def df_change(df: pd.DataFrame) -> pd.DataFrame:
    """moves around the df so it is just the episode description"""
    df = pd.DataFrame(df).rename(columns={'Title': 'Summary'})
    # print(df)
    df = df[df.Summary.apply(lambda x: len(str(x)) > 50)]
    return df[['Summary']]


def scrape(url: str, seasons: int, voy=False) -> pd.DataFrame:
    """gets the html and yields the episode descriptions as a dataframe"""
    # print(url)
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    episodetable = soup.select('table', {'class': 'wikitable'})
    # print(episodetable)
    dfs = pd.read_html(str(episodetable))
    dfs = dfs[1:seasons+1] if seasons > 1 else [dfs[1]]
    if voy:
        dfs = dfs[:2] + dfs[4:]
    for df in dfs:
        # print(df)
        yield df_change(df)


def main():
    filename = 'most_st_wikipages.txt'
    problem_url = 'https://en.wikipedia.org/wiki/List_of_Star_Trek:_Voyager_episodes'
    # scrape(url, 7, True)
    scraped = []
    with open(filename) as f:
        for line in f:
            season, url = line.strip().split()
            voy = True if url == problem_url else False
            print("scraping:", url)
            scraped.append(pd.concat(list(scrape(url, int(season), voy))))
    all_summaries = pd.concat(scraped)
    # print(all_summaries)
    outfile = 'star_trek_episode_summaries.csv'
    print('Writing to', outfile, "...")
    all_summaries.to_csv(outfile, index=False)
    print('done')
    # for x in scraped:
    #     print(x)


if __name__ == '__main__':
    main()
