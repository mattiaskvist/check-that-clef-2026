from scholarly import scholarly
import csv
import time


def scrape_johan_boye_papers():
    # 1. Initialize a proxy to bypass Google's bot detection
    print("Setting up proxy (this might take a moment to find a free one)...")
    # pg = ProxyGenerator()
    # pg.FreeProxies()
    # scholarly.use_proxy(pg)

    # 2. Use Johan Boye's exact Scholar ID to skip the search barrier entirely
    author_id = "Ukmi_9YAAAAJ"
    print(f"Fetching profile for ID: {author_id}...")

    try:
        # Fetch the author directly
        author = scholarly.search_author_id(author_id)
    except Exception as e:
        print(f"Could not fetch author. Google might still be blocking the proxy: {e}")
        return

    print(f"Success! Found: {author.get('name')}")
    print("Filling author profile to retrieve the publication list...")

    # 3. Fill author profile to grab the list of papers
    author = scholarly.fill(author)
    total_pubs = len(author.get("publications", []))
    print(f"Found {total_pubs} publications. Extracting data...")

    output_filename = "scripts/johan_boye_papers.csv"

    # 4. Initialize the CSV file and write the header row
    with open(output_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "Authors", "Venue", "Abstract"])

    # 5. Iterate and extract, saving one by one
    successful_scrapes = 0
    for i, pub in enumerate(author["publications"], 1):
        try:
            # Fill the publication to get the abstract
            pub_filled = scholarly.fill(pub)
            bib = pub_filled.get("bib", {})

            title = bib.get("title", "")
            authors = (bib.get("author", ""),)
            venue = bib.get("venue", bib.get("journal", ""))
            abstract = bib.get("abstract", "")

            print(f"[{i}/{total_pubs}] Scraped: {title}")

            # Append the single row immediately to the CSV
            with open(output_filename, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([title, authors, venue, abstract])

            successful_scrapes += 1

            # Sleep to pace out the requests and avoid triggering blocks
            time.sleep(2)

        except Exception as e:
            print(f"[{i}/{total_pubs}] Error fetching paper: {e}")

    print(
        f"\nDone! Successfully saved {successful_scrapes} out of {total_pubs} papers to {output_filename}"
    )


if __name__ == "__main__":
    scrape_johan_boye_papers()
