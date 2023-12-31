## Objective: Converts High Level BTM List into Tibble Long Format for Convenient Merging with Feature Data

library(tidyverse)
source("https://raw.githubusercontent.com/TranLab/kspzv1-systems/main/helper/Make%20low%20and%20high%20level%20BTMs.R")

#enframe(hilevel.list)
hiBTM.geneid.long <- do.call(rbind, hilevel.list) %>%
  t() %>%
  as_tibble() %>%
  pivot_longer(., cols = 1:ncol(.), names_to = "hiBTM", values_to = "GeneSymbol") %>%
  group_by(hiBTM) %>%
  distinct(GeneSymbol, .keep_all=TRUE) %>%
  ungroup() %>%
  arrange(.,GeneSymbol)
#check
# intersect(foo[foo$`high-level BTM`=="B CELLS",]$GENEID, foo[foo$`high-level BTM`=="T CELLS",]$GENEID)
# setdiff(foo[foo$`high-level BTM`=="B CELLS",]$GENEID, foo[foo$`high-level BTM`=="T CELLS",]$GENEID)
# setdiff(foo[foo$`high-level BTM`=="T CELLS",]$GENEID, foo[foo$`high-level BTM`=="B CELLS",]$GENEID)
# summary(factor(foo[foo$`high-level BTM`=="B CELLS",]$GENEID))
hiBTM.geneid.wide <- hiBTM.geneid.long %>%
  mutate(present = 1) %>%
  pivot_wider(., names_from = hiBTM, values_from = present) %>%
  replace(is.na(.), 0)

#Monaco
#dl "MonacoModules.rds" from google drive
temp <- tempfile(fileext = ".rds")
dl <- drive_download(
  as_id("1-yJW005oPr0RcjXR0YETs9nnRO5g6hYz"), path = temp, overwrite = TRUE)
monaco.list <- readRDS(file = dl$local_path)
monaco.geneid.long <- do.call(rbind, monaco.list) %>%
  t() %>%
  as_tibble() %>%
  pivot_longer(., cols = 1:ncol(.), names_to = "MonacoMods", values_to = "GeneSymbol") %>%
  group_by(MonacoMods) %>%
  distinct(GeneSymbol, .keep_all=TRUE) %>%
  ungroup() %>%
  arrange(.,GeneSymbol)
monaco.geneid.wide <- monaco.geneid.long %>%
  mutate(present = 1) %>%
  pivot_wider(., names_from = MonacoMods, values_from = present) %>%
  replace(is.na(.), 0)
