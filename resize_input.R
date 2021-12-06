library(tidyverse)
infolder = "/pine/scr/t/i/tianyou/Yalin_GAN/data/ADNIHippoCSV/"
outfoler = "/pine/scr/t/i/tianyou/Yalin_GAN/data/resized/"
files = list.files(infolder)

for (f in files){
  orig = read_table2(file.path(infolder, f), col_names = F)
  outf = matrix(NA, nrow = 50 * 75, ncol = 7)
  for (i in 1:7){
    orig_vec = orig[1:15000,i]
    orig_mat = matrix(orig_vec[[1]], nrow = 100, ncol = 150)
    orig_row_1 = orig_mat[seq(1,99,by=2),]
    orig_row_2 = orig_mat[seq(2,100,by=2),]
    orig_row_new = (orig_row_1 + orig_row_2) /2
    orig_col1 = orig_row_new[,seq(1, 149, by=2)]
    orig_col2 = orig_row_new[,seq(2, 150, by=2)]
    orig_col_new = (orig_col1 + orig_col2) / 2
    out_vec = as.vector(orig_col_new)
    outf[, i] = out_vec
  }
  colnames(outf) = colnames(orig)
  outd = as_tibble(outf)
  write_delim(outd, file.path(outfoler, f),delim = " ", col_names = F)
}
