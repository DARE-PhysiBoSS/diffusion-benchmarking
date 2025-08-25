library(ggplot2)
library(cowplot)
library(sitools)
library(viridis)
library(dplyr)


W <- 4.804
H <- 2
S <- 1
point_size <- 0.8
line_size <- 0.5
linecolors <- scale_color_brewer(palette = "Set1")
theme <- theme_cowplot(font_size = 7)

sisec <- Vectorize(
  function(t) {
    if (is.na(t)) {
      NA
    } else {
      sitools::f2si(t / 10^6, "s")
    }
  }
)

domain_label <- function(x) parse(text = paste0(x, "^3"))

ops <- function(n, s) s * n**3

{
  data <- read.csv("baseline.csv", header = TRUE, sep = ",")

  data <- data %>%
    group_by(across(-matches("(init_time)|(time)|(std_dev)"))) %>%
    slice_min(std_dev) %>%
    ungroup()
  data <- data.frame(data)

  ggsave("baseline.pdf",
    device = "pdf", units = "in", scale = S, width = W, height = H,
    ggplot(data, aes(
      x = nx, y = time, color = factor(s), shape = factor(s)
    )) +
      geom_point(size = point_size) +
      geom_line(linewidth = line_size) +
      xlab("Domain size (log-scale)") +
      ylab("Wall-time (log-scale)") +
      scale_color_manual(values = RColorBrewer::brewer.pal(9, "YlGnBu")[2:9]) +
      labs(color = "Substrates", shape = "Substrates") +
      scale_y_log10(labels = sisec) +
      scale_x_log10(labels = domain_label, breaks = c(32, 64, 128, 256)) +
      facet_wrap(~precision) +
      theme +
      background_grid() +
      theme(legend.position = "bottom")
  )

  ggsave("baseline-normalized.pdf",
    device = "pdf", units = "in", scale = S, width = W, height = H,
    ggplot(data, aes(
      x = nx, y = time / ops(nx, s), color = factor(s), shape = factor(s)
    )) +
      geom_point(size = point_size) +
      geom_line(linewidth = line_size) +
      xlab("Domain size (log-scale)") +
      ylab("Time per op (log-scale)") +
      scale_color_manual(values = RColorBrewer::brewer.pal(9, "YlGnBu")[2:9]) +
      labs(color = "Substrates", shape = "Substrates") +
      scale_y_log10(labels = sisec) +
      scale_x_log10(labels = domain_label, breaks = c(32, 64, 128, 256)) +
      facet_wrap(~precision) +
      theme +
      background_grid() +
      theme(legend.position = "bottom")
  )
}

{
  data <- read.csv("transpose_temporal.csv", header = TRUE, sep = ",")

  data <- data %>%
    group_by(across(-matches("(init_time)|(time)|(std_dev)"))) %>%
    slice_min(std_dev) %>%
    ungroup()
  data <- data.frame(data)

  for (val in unique(data$s)) {
    data_sub <- filter(data, s == val)

    ggsave(paste0("transpose_temporal_cont_s", val, ".pdf"),
      device = "pdf", units = "in", scale = S, width = W * 2, height = H * 2,
      ggplot(data_sub, aes(
        x = nx, y = time,
        color = continuous_x_diagonal, shape = continuous_x_diagonal
      )) +
        geom_point(size = point_size) +
        geom_line(linewidth = line_size) +
        xlab("Domain size (log-scale)") +
        ylab("Wall-time (log-scale)") +
        scale_color_manual(values = RColorBrewer::brewer.pal(9, "YlGnBu")[2:9]) +
        labs(color = "Tile Size", shape = "Tile Size") +
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = domain_label, breaks = c(32, 64, 128, 256)) +
        facet_grid(precision ~ x_tile_size) +
        theme +
        background_grid() +
        theme(legend.position = "bottom")
    )

    data_sub <- filter(data_sub, continuous_x_diagonal == "true")

    ggsave(paste0("transpose_temporal_tile_s", val, ".pdf"),
      device = "pdf", units = "in", scale = S, width = W * 2, height = H * 2,
      ggplot(data_sub, aes(
        x = nx, y = time / ops(nx, val),
        color = factor(x_tile_size), shape = factor(x_tile_size)
      )) +
        geom_point(size = point_size) +
        geom_line(linewidth = line_size) +
        xlab("Domain size (log-scale)") +
        ylab("Time-per-op") +
        scale_color_manual(values = RColorBrewer::brewer.pal(9, "YlGnBu")[2:9]) +
        labs(color = "Tile Size", shape = "Tile Size") +
        scale_y_log10(labels = sisec) +
        # scale_x_log10(labels = domain_label, breaks = c(32, 64, 128, 256)) +
        facet_grid(~precision) +
        theme +
        background_grid() +
        theme(legend.position = "bottom")
    )
  }
}

{
  data <- read.csv("partial_blocking.csv", header = TRUE, sep = ",")

  data <- data %>%
    group_by(across(-matches("(init_time)|(time)|(std_dev)"))) %>%
    slice_min(std_dev) %>%
    ungroup()
  data <- data.frame(data)

  for (val in unique(data$s)) {
    data_sub <- filter(data, s == val)

    ggsave(paste0("partial_blocking_cont_s", val, ".pdf"),
      device = "pdf", units = "in", scale = S, width = W * 2, height = H * 2,
      ggplot(data_sub, aes(
        x = nx, y = time / ops(nx, s),
        color = continuous_x_diagonal, shape = continuous_x_diagonal
      )) +
        geom_point(size = point_size) +
        geom_line(linewidth = line_size) +
        xlab("Domain size (log-scale)") +
        ylab("Wall-time (log-scale)") +
        scale_color_manual(values = RColorBrewer::brewer.pal(9, "YlGnBu")[2:9]) +
        labs(color = "Tile Size", shape = "Tile Size") +
        scale_y_log10(labels = sisec) +
        scale_x_log10(labels = domain_label, breaks = c(32, 64, 128, 256)) +
        facet_grid(precision ~ x_tile_size) +
        theme +
        background_grid() +
        theme(legend.position = "bottom")
    )

    data_sub <- filter(data_sub, continuous_x_diagonal == "true")

    ggsave(paste0("partial_blocking_tile_s", val, ".pdf"),
      device = "pdf", units = "in", scale = S, width = W * 2, height = H * 2,
      ggplot(data_sub, aes(
        x = nx, y = time / ops(nx, val),
        color = factor(x_tile_size), shape = factor(x_tile_size)
      )) +
        geom_point(size = point_size) +
        geom_line(linewidth = line_size) +
        xlab("Domain size (log-scale)") +
        ylab("Time-per-op") +
        scale_color_manual(values = RColorBrewer::brewer.pal(9, "YlGnBu")[2:9]) +
        labs(color = "Tile Size", shape = "Tile Size") +
        scale_y_log10(labels = sisec) +
        # scale_x_log10(labels = domain_label, breaks = c(32, 64, 128, 256)) +
        facet_grid(~precision) +
        theme +
        background_grid() +
        theme(legend.position = "bottom")
    )
  }
}


{
  data <- read.csv("full-blocking.csv", header = TRUE, sep = ",")

  data <- data %>%
    group_by(across(-matches("(init_time)|(time)|(std_dev)"))) %>%
    slice_min(std_dev) %>%
    ungroup()
  data <- data.frame(data)

  data$cx <- sapply(strsplit(data$cores_division, ","), `[`, 1)
  data$cy <- sapply(strsplit(data$cores_division, ","), `[`, 2)
  data$cz <- sapply(strsplit(data$cores_division, ","), `[`, 3)

  data <- filter(
    data,
    as.numeric(nx) %% as.numeric(cx) == 0 & as.numeric(ny) %% as.numeric(cy) == 0 & as.numeric(nz) %% as.numeric(cz) == 0 &
      as.numeric(nx) / as.numeric(cx) > 3 & as.numeric(ny) / as.numeric(cy) > 3 & as.numeric(nz) / as.numeric(cz) > 3
  )

  data$scores <- sapply(strsplit(data$cores_division, ","), function(x) {
    paste(sort(as.numeric(x)), collapse = ",")
  })

  for (s_val in unique(data$s)) {
    for (c_val in unique(data$scores)) {
      data_sub <- filter(data, s == s_val & scores == c_val)

      if (dim(data_sub)[1] == 0) {
        next
      }

      ggsave(paste0("full_blocking_s", s_val, "c_", c_val, ".pdf"),
        device = "pdf", units = "in", scale = S, width = W * 3, height = H * 3,
        ggplot(data_sub, aes(
          x = nx, y = time / ops(nx, s), color = factor(cores_division), shape = factor(cores_division)
        )) +
          geom_point(size = point_size) +
          geom_line(linewidth = line_size) +
          xlab("Domain size (log-scale)") +
          ylab("Wall-time (log-scale)") +
          # scale_color_manual(values = RColorBrewer::brewer.pal(9, "YlGnBu")[2:9]) +
          labs(color = "Substrates", shape = "Substrates") +
          scale_y_log10(labels = sisec) +
          scale_x_log10(labels = domain_label, breaks = c(32, 64, 128, 256)) +
          facet_grid(precision ~ sync_step) +
          theme +
          background_grid() +
          theme(legend.position = "bottom")
      )
    }
  }
}

{
  data_baseline <- read.csv("baseline.csv", header = TRUE, sep = ",")
  data_baseline <- subset(data_baseline, select = c(algorithm, precision, dims, iterations, s, nx, ny, nz, init_time, time, std_dev))
  data_temporal <- read.csv("transpose_temporal.csv", header = TRUE, sep = ",")
  data_temporal <- subset(data_temporal, select = c(algorithm, precision, dims, iterations, s, nx, ny, nz, init_time, time, std_dev))
  partial_blocking <- read.csv("partial_blocking.csv", header = TRUE, sep = ",")
  partial_blocking <- subset(partial_blocking, select = c(algorithm, precision, dims, iterations, s, nx, ny, nz, init_time, time, std_dev))
  full_blocking <- read.csv("full-blocking.csv", header = TRUE, sep = ",")
  full_blocking <- subset(full_blocking, select = c(algorithm, precision, dims, iterations, s, nx, ny, nz, init_time, time, std_dev))
  full_blocking_hbm <- read.csv("full-blocking-hbm.csv", header = TRUE, sep = ",")
  full_blocking_hbm <- subset(full_blocking_hbm, select = c(algorithm, precision, dims, iterations, s, nx, ny, nz, init_time, time, std_dev))
  full_blocking_hbm["algorithm"] <- "fb-hbm"

  data <- c()
  data <- rbind(data, data_baseline, data_temporal, partial_blocking, full_blocking, full_blocking_hbm)

  data <- data %>%
    group_by(across(-matches("(init_time)|(time)|(std_dev)"))) %>%
    slice_min(time) %>%
    ungroup()

  data <- subset(data, nx > 32)

  data <- data.frame(data)

  ggsave("all.pdf",
    device = "pdf", units = "in", scale = S, width = W * 2, height = H * 2,
    ggplot(data, aes(
      x = nx, y = time / ops(nx, s), color = algorithm, shape = algorithm
    )) +
      geom_point(size = point_size) +
      geom_line(linewidth = line_size) +
      xlab("Domain size (log-scale)") +
      ylab("Wall-time (log-scale)") +
      scale_color_manual(values = RColorBrewer::brewer.pal(8, "YlGnBu")[3:9]) +
      labs(color = "Substrates", shape = "Substrates") +
      scale_y_log10(labels = sisec) +
      scale_x_log10(labels = domain_label, breaks = c(32, 64, 128, 256)) +
      facet_grid(s ~ precision) +
      theme +
      background_grid() +
      theme(legend.position = "bottom")
  )
}
