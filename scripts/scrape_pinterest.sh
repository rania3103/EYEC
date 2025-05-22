#!/bin/bash

# Scrape images using pinterest via gallery-dl
Quotes
gallery - dl - -range 1 - 100 "https://www.pinterest.com/search/pins/?q=motivational quote instagram post" - d data / quotes
gallery - dl - -range 1 - 100 "https://www.pinterest.com/search/pins/?q=inspirational quote poster" - d data / quotes
gallery - dl - -range 1 - 100 "https://www.pinterest.com/search/pins/?q=life advice typography" - d data / quotes

echo "Scraping done"
