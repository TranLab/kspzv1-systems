#Gene2ModuleExpressionScores2

#Description
#This function collapses a gene expression matrix to module-level expression scores using the within sample method

#Usage
#Gene2ModuleExpressionScores2(gene_expression_dat, module_list = c("lowBTMs", "highBTMs", "BloodGen3Module","MonacoModules"), summary_stat = c(sum, var, sd)) 

#Arguments
#gene_expression_dat    matrix of normalized expression with rownames as HGNC gene symbols or ExpressionSet object containing such a matrix with rownames with HGNC gene symbols within the feature data.
#module_list            name of module set to use. Can be "lowBTMs", "highBTMs", "BloodGen3Module","MonacoModules", "MSigDB_Hallmark_v7.4", or "MSigDB_C2_kegg_v7.4").
#summary_stat           sum, var, or sd
#z_threshold            z score threshold for inclusion of gene within a module based on distance of its expression (in terms of standard deviations) from the mean

#Value
#Output is a dataframe with rownames as module names and column values representing the mean or median or member genes within a module


library(tidyverse)
library(Biobase)
library(curl)
library(data.table)


Gene2ModuleExpressionScores2 <- function(gene_expression_dat,
                                         module_list = c("lowBTMs", "highBTMs", "BloodGen3Module","MonacoModules", "MSigDB_Hallmark_v7.4", "MSigDB_C2_kegg_v7.4"),
                                         summarized_stat = c(sum, var, sd),
                                         z_threshold){
  temp <- tempfile(fileext = ".rds")
  url <- paste0("https://github.com/TranLab/ModuleLists/blob/main/",module_list, ".rds?raw=true")
  myModuleList <- readRDS(url(url, method="libcurl"))
  foo2 <- Biobase::exprs(gene_expression_dat) %>%
    scale() %>% #scale data by column (within sample)
    data.frame() %>%
    setNames(paste0('samples.', names(.))) %>%
    rownames_to_column(var = "rownames") %>%
    left_join(., fData(gene_expression_dat) %>%
                dplyr::select(contains("hgnc") |
                                contains("HGNC") |
                                contains("hugo") |
                                contains("symbol") |
                                contains("Symbol")) %>%
                rownames_to_column(var = "rownames"),
              by = "rownames") %>%
    pivot_longer(cols = contains("samples."), names_to = "sample", values_to = "expression") %>%
    mutate(sample = gsub("samples\\.", "", sample))
  
  df.foo <- myModuleList
  
  for(i in 1:length(myModuleList)){
    df.foo[[i]] <- as_tibble(myModuleList[i]) %>%
      mutate(GeneSymbol = .[[1]]) %>%
      mutate(!!names(myModuleList[i]) := 1) %>%
      dplyr::relocate(., GeneSymbol, .before = !!names(myModuleList[i]))
    }
  
  df.all <- plyr::join_all(df.foo, type = "full", by = "GeneSymbol")
  df.all <- df.all %>%
    replace(is.na(.), 0) %>%
    rename_with(.cols = 2:ncol(.), function(x){paste0("modules.", x)})
  
   myMetric <- summarized_stat
  module.lengths <- myModuleList
  
  for(i in 1:length(myModuleList)){
    module.lengths[[i]]<- length(myModuleList[[i]])
    }
  
  module.lengths.df <- data.frame(module.lengths, check.names = FALSE)
  
  df.all2 <- foo2 %>%
    left_join(., df.all, by = "GeneSymbol") %>%
    as_tibble() %>%
    mutate_at(vars(contains("modules.")) , ~ ifelse(. == 1, expression, 0)) %>% #inclusion: if a gene is included in a module (=1), then use normalized expression, else use 0
    mutate_at(vars(contains("modules.")) , ~ ifelse(abs(.) > z_threshold, expression, 0)) %>% #filter: if absolute value is greater than z_threshold, then use normalized expression, else use 0
    # mutate_at(vars(contains("modules.")) , ~ ifelse(. > z_threshold, 1, .)) %>% #if value is greater than z_threshold, then count (use 1), else use existing value
    # mutate_at(vars(contains("modules.")) , ~ ifelse(. < z_threshold*(-1), -1, .)) %>% #if value is less than z_threshold*-1, then count (use -1), else use existing value
    replace(is.na(.), 0) %>%
    dplyr::select(sample, GeneSymbol, contains("modules.")) %>%
    dplyr::group_by(sample) %>%
    summarise_at(.vars = names(.)[3:ncol(.)],
                   .funs = myMetric*sum)

  rownames(df.all2) <- NULL
  mes_score <- df.all2 %>%
    column_to_rownames('sample') %>%
    rename_with(.cols = everything(), function(x){gsub("modules\\.", "",x)})
  
  mes_score <- mes_score[,colnames(module.lengths.df)]
  
  if(all(colnames(module.lengths.df) == colnames(mes_score))){
    for(j in 1:nrow(mes_score)){
      mes_score[j,] <- mes_score[j,]/module.lengths.df
    }
  }
  mes_score <- t(mes_score[, colSums(mes_score != 0) > 0]) %>%
    as.data.frame()
}
