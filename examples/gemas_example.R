# Code to reproduce the results of Sipilä et al. (2023).

# packages
library("SpatialBSS")
library("StatDA")
library("ggmap")
library("sf")
library("tidyr")
library("xtable")
library("ggplot2")
library("NonlinearBSS")
library("kernelshap")
devtools::install_github("mikasip/NonlinearBSS")
library("NonlinearBSS")

# functions
contras_mat <- function(x) {
    c_mat <- matrix(0, nrow = ncol(x), ncol = ncol(x) - 1)
    for (i in 1:ncol(c_mat)) {
        c_mat[1:i, i] <- 1 / i
        c_mat[i + 1, i] <- (-1)
        c_mat[, i] <- c_mat[, i] * sqrt(i / (i + 1))
    }
    return(c_mat)
}

clr_func <- function(x) {
    x_clr <- apply(x, 1, function(row) row / prod(row)^(1 / length(row)))
    x_clr <- log(x_clr)
    return(t(x_clr))
}

plot_map <- function(coords, variable, map, quant = TRUE) {
    x <- data.frame(x = coords[, 1], y = coords[, 2], variable = variable)

    if (quant) {
        q <- quantile(variable, c(0, 0.05, 0.25, 0.75, 0.95, 1))
        leg.round <- 2
        leg.wid <- 4
        leg.just <- "right"
        leg <- rep(NA, length(q) - 1)
        leg[1] <- paste(
            "  ", roundpretty(q[1], leg.round),
            "-", format(roundpretty(q[1 + 1], leg.round),
                width = leg.wid, justify = leg.just
            )
        )
        for (i in 2:(length(q) - 1)) {
            leg[i] <- paste(
                ">", roundpretty(q[i], leg.round),
                "-", format(roundpretty(q[i + 1], leg.round),
                    width = leg.wid, justify = leg.just
                )
            )
        }

        x$variable <- cut(variable,
            breaks = q,
            include.lowest = TRUE
        )

        shapes <- rev(c(17, 17, 20, 15, 15))
        shape_size <- rev(c(1.7, 1.0, 0.3, 1.2, 1.8))
        shape_size <- 1.6 * rev(c(1.3, 0.6, 0.2, 0.9, 1.5))
        color <- c("royalblue3", "royalblue1", "black", "brown1", "brown3")

        g <- ggmap::ggmap(map,
            base_layer = ggplot(aes(
                x = x, y = y, color = variable, shape = variable,
                size = variable
            ), data = x)
        ) +
            geom_point(alpha = 1) +
            theme_bw() +
            xlab(expression("Longitude" * ~ degree * W)) +
            ylab(expression("Latitude" * ~ degree * N)) +
            theme(legend.position = "bottom") +
            scale_shape_manual(
                values = shapes,
                labels = leg
            ) +
            scale_size_manual(
                values = shape_size,
                labels = leg
            ) +
            scale_color_manual(
                values = color,
                labels = leg
            ) +
            guides(shape = guide_legend(nrow = 2, byrow = TRUE)) +
            theme(legend.title = element_blank()) +
            theme(legend.text = element_text(size = 11))
    } else {
        g <- ggmap::ggmap(map,
            base_layer = ggplot(aes(x = x, y = y, color = variable), data = x)
        ) +
            geom_point(alpha = 1) +
            theme_bw() +
            xlab(expression("Longitude" * ~ degree * W)) +
            ylab(expression("Latitude" * ~ degree * N)) +
            theme(legend.position = "bottom") +
            # scale_colour_gradientn(colours =rainbow(7))	+
            # scale_colour_gradientn(low = "grey", high = "brown") +
            scale_colour_gradientn(colors = c("turquoise", "yellow", "red", "black")) +
            theme(legend.title = element_blank()) +
            theme(legend.text = element_text(size = 11))
    }
    plot(g)
    return(g)
}

# data
library("robCompositions")
data("gemas")
head(gemas)
gemas <- gemas[-which(gemas$Ycoord == min(gemas$Ycoord)), ]
coords <- as.matrix(gemas[, c("Xcoord", "Ycoord")])
row.names(coords) <- NULL

coords_sf <- sf::st_as_sf(data.frame(coords),
    coords = c(1, 2)
)
st_crs(coords_sf) <- "EPSG:3035"
coords_sf <- st_transform(coords_sf, "+proj=longlat +datum=WGS84")
coords_ll <- st_coordinates(coords_sf, "+proj=longlat +datum=WGS84")
head(coords_ll)

aux_grid_start <- c(min(coords[, 1]), min(coords[, 2]))
size <- 100000
aux_grid_x <- seq(aux_grid_start[1] - size * 5, max(coords[, 1]) + size * 5, by = size)
aux_grid_y <- seq(aux_grid_start[2] - size * 5, max(coords[, 2]) + size * 5, by = size)
grid_coords <- data.frame(matrix(NA, ncol = 4))
colnames(grid_coords) <- c("x", "y", "order", "grid_ind")
ind <- 1
grid_ind <- 1
for (i in 1:(length(aux_grid_x) - 1)) {
    for (j in 1:(length(aux_grid_y) - 1)) {
        grid_coords[ind, ] <- c(aux_grid_x[i], aux_grid_y[j], 1, grid_ind)
        grid_coords[ind + 1, ] <- c(aux_grid_x[i], aux_grid_y[j + 1], 2, grid_ind)
        grid_coords[ind + 2, ] <- c(aux_grid_x[i + 1], aux_grid_y[j + 1], 3, grid_ind)
        grid_coords[ind + 3, ] <- c(aux_grid_x[i + 1], aux_grid_y[j], 4, grid_ind)
        grid_ind <- grid_ind + 1
        ind <- ind + 4
    }
}
grid_coords_sf <- st_as_sf(data.frame(grid_coords),
    coords = c(1, 2)
)
st_crs(grid_coords_sf) <- "EPSG:3035"
grid_coords_sf <- st_transform(grid_coords_sf, "+proj=longlat +datum=WGS84")
grid_coords_ll <- st_coordinates(grid_coords_sf, "+proj=longlat +datum=WGS84")
head(grid_coords_ll)
grid_coords$lon <- grid_coords_ll[, 1]
grid_coords$lat <- grid_coords_ll[, 2]

# register_stadiamaps("YOUR_API_KEY", write = TRUE)
kola_map <- get_stadiamap(c(left = min(coords_ll[, 1]) - 0.5, bottom = min(coords_ll[, 2]) - 0.5, right = max(coords_ll[, 1]) + 0.5, top = max(coords_ll[, 2]) + 0.5),
    maptype = "stamen_terrain_background", zoom = 4
)

g <- ggmap(kola_map,
    base_layer = ggplot(
        data = data.frame(coords_ll),
        aes(y = Y, x = X)
    )
) + geom_point(color = "darkred") +
    geom_polygon(aes(lon, lat, group = grid_ind), alpha = 0, linewidth = 0.1, colour = "black", data = grid_coords)
plot(g)

vars <- c("Al", "Ba", "Ca", "Cr", "Fe", "K", "Mg", "Mn", "Na", "Nb", "P", "Si", "Sr", "Ti", "V", "Y", "Zn", "Zr")

colnames(gemas)
gemas_dat <- as.matrix(gemas[, vars])
gemas_df <- data.frame(gemas_dat)
gemas_dat <- as.matrix(gemas_df)
row.names(gemas_dat) <- NULL

clr_dat <- clr_func(gemas_dat)
c_mat <- contras_mat(gemas_dat)

data_ilr <- log(gemas_dat) %*% c_mat

p <- 17
seed <- 11092023
resiVAE <- iVAE_spatial(data_ilr, coords, c(100000, 100000), c(1, 1), p, epochs = 1000, seed = seed, batch_size = 64)

ic_idx <- 15
g <- plot_map(coords_ll, clr_dat[, 2], map = kola_map, quant = FALSE)
ggsave(plot = g, paste0("ic_", ic_idx, ".pdf"), height = 6, width = 4.9)

select_points_sparsely <- function(coords, n) {
    distance_matrix <- as.matrix(dist(coords))
    selected_inds <- numeric(n)
    selected_inds[1] <- 1
    for (i in 2:(n)) {
        dists_unselected_to_selected <- as.matrix(distance_matrix[-selected_inds, selected_inds])
        row_min_dists <- apply(dists_unselected_to_selected, 1, min)
        max_dist_ind <- which(row_min_dists == max(row_min_dists))
        selected_inds[i] <- as.numeric(names(max_dist_ind)[1])
    }
    return(list(coords = coords[selected_inds, ], inds = selected_inds))
}
# Interpretations of ICs

X <- cbind(clr_dat, resiVAE$aux_data)
X <- as.data.frame(X)
bg_points <- select_points_sparsely(coords, 200)

g_bg <- ggmap(kola_map,
    base_layer = ggplot(
        data = data.frame(coords_ll[bg_points$inds, ]),
        aes(y = Y, x = X)
    )
) + geom_point(color = "darkred") + xlab("Longitude") + ylab("Latitude")
plot(g_bg)

bg_X <- X[bg_points$inds, ]
explainer <- kernelshap(resiVAE, X,
    bg_X = bg_X, orig_dim = 18, pred_fun = function(object, X, orig_dim) {
        ilr_dat <- as.matrix(X[, 1:18]) %*% c_mat
        predict(object, newdata = as.matrix(ilr_dat), aux_data = as.matrix(X[, (orig_dim + 1):ncol(X)]))
    }, feature_names = colnames(X)[1:18]
)
orig_dim <- 17

mashap <- data.frame(matrix(NA, ncol = 17, nrow = 18))
rownames(mashap) <- sapply(colnames(gemas_dat), function(name) paste0("clr(", name, ")"))
colnames(mashap) <- sapply(1:17, function(i) paste0("IC", i))

i <- 1
for (l in explainer$S) {
    mashap[, i] <- apply(abs(l), 2, mean)
    i <- i + 1
}
mashap_scaled <- sweep(mashap, 2, colSums(mashap), "/")

X <- as.data.frame(resiVAE$IC_unscaled)
X <- X
bg_X <- X[bg_points$inds, ]
explainer2 <- kernelshap(resiVAE, X, bg_X = bg_X, pred_fun = function(object, X) {
    pred <- predict.iVAE(object, newdata = as.matrix(X), IC_to_data = TRUE)
    return(as.matrix(pred) %*% t(c_mat))
})

explainer2$baseline
mashap2 <- data.frame(matrix(NA, ncol = 17, nrow = 18))
rownames(mashap2) <- sapply(colnames(gemas_dat), function(name) paste0("clr(", name, ")"))
colnames(mashap2) <- sapply(1:17, function(i) paste0("IC", i))
i <- 1
for (l in explainer2$S) {
    mashap2[i, ] <- apply(abs(l), 2, mean)
    i <- i + 1
}
mashap_scaled2 <- sweep(mashap2, 1, rowSums(mashap2), "/")
mashap_scaled2
importance_vals <- colMeans(mashap_scaled2)
xtable(mashap_scaled2, digits = 3)
xtable(data.frame(t(colMeans(mashap_scaled2))), digits = 3)
sorted_mashap_scaled2 <- mashap_scaled2[, order(importance_vals, decreasing = TRUE)]
xtable(sorted_mashap_scaled2, digits = 3)
xtable(data.frame(t(colMeans(sorted_mashap_scaled2))), digits = 3)

sorted_mashap_scaled <- mashap_scaled[, order(importance_vals, decreasing = TRUE)]
xtable(sorted_mashap_scaled, digits = 3)
