library(tidyverse)
folder = "/pine/scr/t/i/tianyou/Yalin_GAN/data/ADNIHippoCSV/"
files = list.files(folder)
set.seed(16)

png("real_samples.png", width = 800, height = 800)
par(mar=c(1,1,1,1))
layout(matrix(1:16,nr=4,byr=T))
files_sel = files[sample.int(length(files), size = 16)]
for (f in files_sel){
  real = read_table2(paste0(folder, f), col_names = F)
  real_mat = matrix(real$X1[1:15000], nrow = 100, ncol = 150)
  image(real_mat)
}
dev.off()



fake = read_csv("samp_epoch4960.csv")
par(mar=c(1,1,1,1))
layout(matrix(1:16,nr=4,byr=T))
for (i in 1:nrow(fake)){
  fake_mat = matrix(as.numeric(fake[i,]), nrow = 100, 
                    ncol = 150, byrow = T)
  #fake_mat[fake_mat>14]=0
  image(fake_mat)
}
