setwd("~/Documents/Weizmann/MOmaps/RNA_decay/SLAMSEQ")

library(grandR)
library(ggplot2)
library(patchwork)
library(reshape2)
library(ggpubr)
library(ggrastr)
library(Cairo)
library(dplyr)
library(tidyr)

# Read grand-slam main output table and rename columns for compatibility with grandR design
dNLS <- read.delim('TDP43dNLS_trim20b.tsv', header = TRUE, check.names = FALSE)
colnames(dNLS) <- gsub('/home/projects/hornsteinlab/welmoed/slamseq/241006_data/dedup_bams/', '', colnames(dNLS))
colnames(dNLS) <- gsub('_', '.', colnames(dNLS))
colnames(dNLS) <- gsub('hr', 'h', colnames(dNLS))
write.table(dNLS, 'TDP43dNLS_trim20b_edited.tsv', quote=FALSE, sep='\t')

# Create the grandR object
dNLS_grand <- ReadGRAND('TDP43dNLS_trim20b_edited.tsv', design = c(Design$Condition,Design$dur.4sU,Design$Replicate),
                  read.percent.conv = TRUE)
Coldata(dNLS_grand)
dNLS_grand

# Filter lowly expressed genes (in at least half of the columns)
dNLS_grand <- FilterGenes(dNLS_grand,minval = 10, mincol = 9) 
dNLS_grand

# PCA
PlotPCA(dNLS_grand, aest=aes(color=duration.4sU.original,shape=Condition))

# Normalize
dNLS_grand <- Normalize(dNLS_grand)
#Default will perform DESeq2 normalization by library size (https://grandr.erhard-lab.de/reference/Normalize.html)
# and (https://rdrr.io/bioc/DESeq2/man/estimateSizeFactorsForMatrix.html)

# Cell viability check
Plot4sUDropoutRankAll(dNLS_grand)

### Kinetics ###
# One gene example
PlotGeneProgressiveTimecourse(dNLS_grand,"STMN2") 
PlotGeneGroupsBars(dNLS_grand,'STMN2')

# All genes
SetParallel(cores = 2)
dNLS_grand <- FitKinetics(dNLS_grand,name = "kinetics")
dNLS_grand

# Get dataframe with half-life values
df <- GetAnalysisTable(dNLS_grand, columns = 'Half-life', by.rows = TRUE)
head(df)
df$Analysis <- gsub("kinetics\\.", "", df$Analysis)
colnames(df) <- gsub('Analysis', 'Condition', colnames(df))
colnames(df) <- gsub('Half-life', 'Half_life', colnames(df))
df$Condition <- factor(df$Condition, levels = c("uninduced", "dox"))

# Add information about P-body-enrichment
PB <- read.csv('Hubstenberger_Pbody_RNAs.csv', header = TRUE)
colnames(PB) <- gsub('Associated.Gene.Name', 'Gene', colnames(PB))
PB <- PB %>% filter(PB$FDR < 0.05)
PB <- PB %>% filter(PB$log2FC > 1)
df <- df %>% mutate(Gene_Type = ifelse(Symbol %in% PB$Gene, "P-body gene", "other"))

# Add information about TDP-43 gene targets
tdp <- read.csv('~/Documents/Weizmann/TDP-43/Gene_lists/TARDBP_POSTAR3_CLIPdb_module_RBP_binding_sites.csv')
length(unique(tdp$Target.gene.symbol))
# filter for more than x binding sites
tdp <- tdp %>% filter(tdp$Binding.site.records > 10) #STMN2 is 17 so above 10
length(unique(tdp$Target.gene.symbol))
df <- df %>% mutate(TDP43_target = ifelse(Symbol %in% tdp$Target.gene.symbol, "TDP-43 target", "other"))

# Separate columns per condition
new_df <- df %>%
  pivot_wider(names_from = Condition, values_from = Half_life, 
              names_prefix = "half_life_") %>%
  rename(uninduced_half_life = half_life_uninduced, 
         dox_half_life = half_life_dox)

write.table(new_df, 'TDP43dNLS_trim20b_half-life.tsv', quote=FALSE, sep='\t')

# Calculate the average half-life for each condition
average_half_life <- df %>%
  group_by(Condition) %>%
  summarize(Average_Half_life = mean(Half_life, na.rm = TRUE))
print(average_half_life)

t.test(Half_life ~ Condition, data = df)

# Perform the Kolmogorov-Smirnov test
ks_test <- ks.test(df$Half_life[df$Condition == "dox"], 
                   df$Half_life[df$Condition == "uninduced"])
ks_test$p.value

# Plot cumulative plot comparing conditions
ggplot(df,aes(x = Half_life,color=Condition))+
  geom_density(alpha = 0.5) +
  scale_x_log10() +
  annotate("text", x = 5, y = 0.9, label = paste("K-S p-value:", format(ks_test$p.value, digits = 3)), size = 5) +
  labs(title = "ECDF Plot with Kolmogorov-Smirnov Test", x = "Half-life (hours)", 
       y = "Density") +
  theme_minimal()

# Plot violin plot with statistical comparison
ggplot(df, aes(x = Condition, y = Half_life, fill = Condition)) +
  geom_violin(trim = FALSE) +
  geom_boxplot(width = 0.1, position = position_dodge(width = 0.9)) +  # Add a boxplot inside the violin plot
  stat_compare_means(method = "t.test") +  # Add statistical comparison (t-test)
  labs(title = "Comparison of Half-life: Uninduced vs Dox",
       x = "Condition", 
       y = "Half-life (hours)") +
  theme_minimal()

# Boxplot
ggplot(df, aes(x = Condition, y = Half_life, fill = Condition)) +
  geom_boxplot(width = 0.4, outliers = TRUE, position = position_dodge(width = 0.9)) + 
  #coord_cartesian(ylim=c(0,50)) +
  stat_summary(fun = mean, geom = "text", aes(label = round(..y.., 2)), 
               vjust = 1, color = "black") +  # Display mean as text
  #stat_compare_means(method = "t.test") +  # Add statistical comparison (t-test)
  labs(title = "Comparison of Half-life: Uninduced vs Dox",
       x = "Condition", 
       y = "Half-life (hours)") +
  theme_minimal()

# Boxplot
ggplot(df, aes(x = Gene_Type, y = Half_life, fill = Gene_Type)) +
  geom_boxplot(width = 0.4, outliers = TRUE, position = position_dodge(width = 0.9)) + 
  #coord_cartesian(ylim=c(0,50)) +
  stat_summary(fun = mean, geom = "text", aes(label = round(..y.., 2)), 
               vjust = 1, color = "black") +  # Display mean as text
  stat_compare_means(method = "t.test") +  # Add statistical comparison (t-test)
  labs(x = "Condition", y = "Half-life (hours)") +
  theme_minimal()
