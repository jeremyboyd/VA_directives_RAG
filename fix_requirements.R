# Author: Jeremy Boyd (jeremy.boyd@va.gor)
# Description: Fix requirements file.

# Packages
library(tidyverse)

# Table of required python packages
req <- read_csv("requirements.txt") |>
    filter(row_number() %in% 5:260) |>
    rename(col = `# This file may be used to create an environment using:`) |>
    mutate(col = str_remove(col, "=.+")) |>
    pull(col) |>
    paste0(collapse = "\n")

# Write to text
cat(req, file = "requirements.txt")
