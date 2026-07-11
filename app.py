## imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import warnings
warnings.simplefilter(action="ignore", category="SettingWithCopyWarning")
import streamlit as st

warnings.warn = original_warn
warnings.warn = lambda *args, **kwargs: None

## streamlit page configuration
st.set_page_config(page_title="Car Auction Analysis", layout="wide")

## title
st.title("Analysis of car auctions in the US")

### header-1
st.markdown(body="## 1. Problem: Where could a used Ford F150 pickup truck with the optimum price-condition state be bought in the US?", unsafe_allow_html=True)

## header-2
st.markdown(body="### 1.1. Profiling of used car auction data", unsafe_allow_html=True)

## dataframe
# dataframe
data_link = st.secrets["car_prices"]
@st.cache_data
def load_data(data_link):
    return pd.read_parquet(data_link)
cars= load_data(data_link)
st.dataframe(data=cars.head())

# data description
with st.expander(label="Data description:"):
	st.dataframe(data=cars.describe())

## global settings
# plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style("darkgrid")
with st.expander(label="Code | Global settings:"):
	st.code("""sns.set_style("darkgrid")""")

## header-2
st.markdown("""### 1.2. Get the relationship matrix of numeric columns with a pairplot, on selected body types""", unsafe_allow_html=True)

## sampled data
cars_sampled = cars.sample(n=50000, random_state=42, replace=False, axis=0)
with st.expander(label="Code | Sampled data:"):
	st.code("""cars_sampled = cars.sample(n=50000, random_state=42, replace=False, axis=0)""")



## graph-1 pairplot: comparison of numeric columns
style_list = ["SUV", "Sedan", "Convertible", "Coupe"]
@st.cache_data
def pairplot():
	return sns.pairplot(
	data=cars_sampled.loc[cars_sampled["body"].isin(style_list), :],
	hue="body",
	palette="husl",
	aspect=1.25,
	dropna=True,
	corner=True,
	diag_kws={"bw_adjust":0.8}
)
g = pairplot()
g.fig.suptitle("Comparison of Numeric Columns", fontsize=20, fontweight="bold")

g.legend.remove()
g.add_legend(loc='upper right',
			 bbox_to_anchor=(0.95, 0.95),
			 title="Body",
			 fontsize=12,
			 frameon=True,
			 facecolor="grey",
			 edgecolor="black")
with st.expander(label="Code | Pair plot, comparison of numeric columns:"):
	st.code("""
	style_list = ["SUV", "Sedan", "Convertible", "Coupe"]
	g = sns.pairplot(
		data=cars_sampled.loc[cars_sampled["body"].isin(style_list), :],
		hue="body",
		palette="husl",
		# kind="reg",
		aspect=1.25,
		dropna=True,
		corner=True,
		diag_kws={"bw_adjust":0.8}
	)
	g.fig.suptitle("Comparison of Numeric Columns", fontsize=20, fontweight="bold")
	g.legend.remove()
	g.add_legend(loc='upper right',
				 bbox_to_anchor=(0.95, 0.95),
				 title="Body",
				 fontsize=12,
				 frameon=True,
				 facecolor="grey",
				 edgecolor="black")
	""")
with st.container(height=500):
	st.pyplot(fig=g, width=400, use_container_width=False)

# header -3
st.markdown("""
### 1.2.1. Findings
* When the year gets closer to the present day, the selling price and estimated price (mmr) increases.
* When condition of the car gets better, the selling price and estimated price (mmr) increases.
* When the distance covered by car (odometer) increases, the selling price and estimated price (mmr) decreases.
* The selling price and estimated price (mmr) are closely linked.
""", unsafe_allow_html=True)



## header - 2
st.markdown("""
## 1.3. Correlation heatmap between numeric variables
""", unsafe_allow_html=True)

## graph-2 heatmap: correlation of numeric columns
cars_corr = cars.corr(numeric_only=True)
fig, ax = plt.subplots()
sns.heatmap(
	data=cars_corr,
	mask=(cars_corr > -0.5) & (cars_corr < 0.5),
	center=0.5,
	square=True,
	annot=True,
	fmt=".2f",
	annot_kws={"family":"sans serif", "fontsize":9},
	cbar=True,
	cbar_kws={"label":"scale", "ticks":(-0.5, 0, 0.5), "fraction":0.1},
	lw=1.5,
	ec="0.5",
	cmap="coolwarm",
	shading="flat",
	snap=True,
	norm="linear",
	ax=ax
)
ax.set_title("Correlation of Numeric Columns", fontsize=14, fontweight="bold")

with st.expander(label="Code | Pair plot, correlation of numeric columns:"):
	st.code("""
cars_corr = cars.corr(numeric_only=True)
fig, ax = plt.subplots()
sns.heatmap(
	data=cars_corr,
	mask=(cars_corr > -0.5) & (cars_corr < 0.5),
	center=0.5,
	square=True,
	annot=True,
	fmt=".2f",
	annot_kws={"family":"sans serif", "fontsize":9},
	cbar=True,
	cbar_kws={"label":"scale", "ticks":(-0.5, 0, 0.5), "fraction":0.1},
	lw=1.5,
	ec="0.5",
	cmap="coolwarm",
	shading="flat",
	snap=True,
	norm="linear",
	ax=ax
)
ax.set_title("Correlation of Numeric Columns", fontsize=14, fontweight="bold")
	""")
with st.container(height=500):
	st.pyplot(fig=fig)

# header -3
st.markdown("""
### 1.3.1. Findings (not clearly visible in the pairplot)
* Year and condition is somewhat positively correlated.
* Odometer values and condition are somewhat negatively correlated.
* Odometer values and year are highly negatively correlated.
""", unsafe_allow_html=True)



## header -2
st.markdown("""
## 1.4. Let's see the negative correlation between odometer and selling price more clearly for selected makes
""", unsafe_allow_html=True)

## graph-3 lmplot: selling price by odometer
brand_list = ['Ford', 'BMW', 'Toyota', 'Chevrolet']
g = sns.lmplot(
	x="odometer",
	y="sellingprice",
	data=cars.query("make in @ brand_list and odometer < 200000").sample(n=1000, random_state=42, axis=0),
	hue="make",
	palette="husl",
	legend="full",
	truncate=True,
	y_jitter=True,
	x_estimator=None,
	# ci=95,
	scatter_kws={"alpha":0.3},
	aspect=1.25
)
plt.xticks(rotation=90)
g.fig.suptitle("Selling Price by Odometer (odometer < 200000)", fontsize=14, fontweight="bold", x=0.38, y=1.03)

g.legend.remove()
g.add_legend(loc='upper right',
			 bbox_to_anchor=(0.63, 0.95),
			 title="Make",
			 fontsize=10,
			 frameon=True,
			 # facecolor="0.9",
			 edgecolor="black",
			 ncol=4,
			 labelspacing=0.1
			)

with st.expander(label="Code | Implot, selling price by odometer:"):
	st.code("""
brand_list = ['Ford', 'BMW', 'Toyota', 'Chevrolet']
g = sns.lmplot(
	x="odometer",
	y="sellingprice",
	data=cars.query("make in @ brand_list and odometer < 200000").sample(n=1000, random_state=42, axis=0),
	hue="make",
	palette="husl",
	legend="full",

	# robust=True,
	truncate=True,
	y_jitter=True,
	x_estimator=None,
	# ci=95,
	scatter_kws={"alpha":0.3},
	aspect=1.25
)
plt.xticks(rotation=90)
g.fig.suptitle("Selling Price by Odometer (odometer < 200000)", fontsize=14, fontweight="bold", x=0.38, y=1.03)

g.legend.remove()
g.add_legend(loc='upper right',
			 bbox_to_anchor=(0.63, 0.95),
			 title="Make",
			 fontsize=10,
			 frameon=True,
			 # facecolor="0.9",
			 edgecolor="black",
			 ncol=4,
			 labelspacing=0.1
			)
	""")
with st.container(height=500):
	st.pyplot(fig=g)

# header -3
st.markdown("""
### 1.4.1. Findings
* BMW is the one that losts its value the most as odometer increases.
* The other three have nearly the same slope.
""", unsafe_allow_html=True)



## header -2
st.markdown("""
## 1.5. Let's see the negative correlation between condition and selling price more clearly
""", unsafe_allow_html=True)

## graph-4 barplot: selling price by condition
bins = np.linspace(1, 5, 9)
labels = ["1-1.5", "1.5-2", "2-2.5", "2.5-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5"]

cars["condition_bins"] = pd.cut(x=cars["condition"], bins=bins, labels=labels, right=True, include_lowest=True)

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(
	x="condition_bins",
	y="sellingprice",
	data=cars,
	hue="condition_bins",
	palette=["r","r","r","r","r","b","b","b"],
	estimator=np.mean,
	errorbar=('ci', 95),
	gap=0.3,
	dodge=False,
	ec="0.0",
	lw=1,
	width=1,
	alpha=0.9,
	ax=ax
)
lows = patches.Patch(color="r", label="condition < 3.5")
highs = patches.Patch(color="b", label="condition > 3.5")
ax.legend(handles=[lows, highs])

ax.set_ylabel("Selling Price")
ax.set_xlabel("Condition Bins")
ax.set_title("Selling Price by Condition", fontsize=14, fontweight="bold", x=0.5, y=1.03)

with st.expander(label="Code | Implot, selling price by condition: "):
	st.code("""
bins = np.linspace(1, 5, 9)
labels = ["1-1.5", "1.5-2", "2-2.5", "2.5-3", "3-3.5", "3.5-4", "4-4.5", "4.5-5"]

cars["condition_bins"] = pd.cut(x=cars["condition"], bins=bins, labels=labels, right=True, include_lowest=True)

fig, ax = plt.subplots(figsize=(10,6))
sns.barplot(
	x="condition_bins",
	y="sellingprice",
	data=cars,
	hue="condition_bins",
	palette=["r","r","r","r","r","b","b","b"],
	estimator=np.mean,
	errorbar=('ci', 95),
	gap=0.3,
	dodge=False,
	ec="0.0",
	lw=1,
	width=1,
	alpha=0.9,
	ax=ax
)
lows = patches.Patch(color="r", label="condition < 3.5")
highs = patches.Patch(color="b", label="condition > 3.5")
ax.legend(handles=[lows, highs])

ax.set_ylabel("Selling Price")
ax.set_xlabel("Condition Bins")
ax.set_title("Selling Price by Condition", fontsize=14, fontweight="bold", x=0.5, y=1.03)
	""")
with st.container(height=500):
	st.pyplot(fig=fig)

# header -3
st.markdown("""
### 1.5.1. Findings
* Findings from the pairplot is confirmed.
""", unsafe_allow_html=True)



## header -2
st.markdown("""
## 1.6. Let's see the relationship between condition and make: categorical variable relations
""", unsafe_allow_html=True)

## graph-5 barplot: selling price by common makes and condition
top_10_brands = cars["make"].value_counts().head(10).index
cars_pivot =(cars
	.pivot_table(index="make", columns="condition_bins", values="sellingprice", aggfunc="mean", observed=True)
	.loc[lambda x: x.index.isin(top_10_brands)]
			)

fig, ax = plt.subplots()
sns.heatmap(
	data=cars_pivot,
	square=False,
	annot=True,
	fmt=".0f",
	annot_kws={"fontsize":8},
	cbar=True,
	cbar_kws={"label":"scale", "fraction":0.12},
	lw=1,
	cmap="coolwarm",
	ax=ax
)
ax.tick_params(labelsize=8)
ax.set_ylabel("Make")
ax.set_xlabel("Condition Bins")
ax.set_title("Selling Price by Common Makes and Condition", fontsize=14, fontweight="bold", x=0.5, y=1.03)

with st.expander(label="Code | Heatmap, selling price by common makes and condition: "):
	st.code("""
top_10_brands = cars["make"].value_counts().head(10).index
cars_pivot =(cars
	.pivot_table(index="make", columns="condition_bins", values="sellingprice", aggfunc="mean", observed=True)
	.loc[lambda x: x.index.isin(top_10_brands)]
			)

fig, ax = plt.subplots()
sns.heatmap(
	data=cars_pivot,
	square=False,
	annot=True,
	fmt=".0f",
	annot_kws={"fontsize":8},
	cbar=True,
	cbar_kws={"label":"scale", "fraction":0.12},
	lw=1,
	cmap="coolwarm",
	ax=ax
)
ax.tick_params(labelsize=8)
ax.set_ylabel("Make")
ax.set_xlabel("Condition Bins")
ax.set_title("Selling Price by Common Makes and Condition", fontsize=14, fontweight="bold", x=0.5, y=1.03)
	""")
with st.container(height=500):
	st.pyplot(fig=fig)

# header -3
st.markdown("""
### 1.6.1. Findings
* Obviously Hyundai, Kia and Nissan models are possible to be bought for respectively cheaper prices and in better conditions.
* BMW models, as envisaged by lmplot above, are getting expensive as the condition gets better.
* Prices of all the other models range between 19 and 20 thousands for the best condition. Therefore, the Ford models could be bought for around 20000.
""", unsafe_allow_html=True)



## header -2
st.markdown("""
## 1.7. Let's dive into the model F150s, first by finding distribution of their selling prices
""", unsafe_allow_html=True)

## graph-6 histplot: selling price distribution (probability)
f150s = cars[cars["model"] == "F-150"]

fig, ax = plt.subplots()
sns.histplot(
	x="sellingprice",
	data=f150s,
	kde=True,
	kde_kws={"bw_adjust":0.1, "cut":True},
	stat="probability",
	ec="black",
	color="orange",
	lw=1,
	binwidth=5000,
	alpha=0.5,
	ax=ax
)
xticks = np.linspace(0, 65000, 14)
ax.set_xticks(ticks=xticks, labels=xticks, rotation=45)
ax.set_xlabel("Selling Price")
ax.set_title("Selling Price Distribution (probability)", fontsize=14, fontweight="bold", x=0.5, y=1.03)

with st.expander(label="Code | Histogram, selling price distribution (probability): "):
	st.code("""
f150s = cars[cars["model"] == "F-150"]

fig, ax = subplots()
sns.histplot(
	x="sellingprice",
	data=f150s,
	kde=True,
	kde_kws={"bw_adjust":0.1, "cut":True},
	stat="probability",
	ec="black",
	color="orange",
	lw=1,
	binwidth=5000,
	alpha=0.5,
	ax=ax
)
xticks = np.linspace(0, 65000, 14)
ax.set_xticks(ticks=xticks, labels=xticks, rotation=45)
ax.set_xlabel("Selling Price")
ax.set_title("Selling Price Distribution (probability)", fontsize=14, fontweight="bold", x=0.5, y=1.03)
	""")
with st.container(height=500):
	st.pyplot(fig=fig)

# header -3
st.markdown("""
### 1.7.1. Findings
* 20% (majority) of the F150s sales are done at 20000 thousands. 
""", unsafe_allow_html=True)



# #header -2
st.markdown("""
## 1.8. Secondly, by comparing of their selling prices between their trims
""", unsafe_allow_html=True)

## graph-7 barplot: selling price of Ford F150 by Trim
fig, ax = plt.subplots()
sns.barplot(
	x="trim",
	y="sellingprice",
	data=f150s,
	hue="trim",
	palette="husl",
	legend=False,
	gap=0.2,
	estimator=np.mean,
	errorbar="se",
	capsize=0.1,
	ec="0.0",
	lw=0.2,
	alpha=0.8,
	ax=ax
)
plt.xticks(rotation=45)
ax.set_ylabel("Selling Price")
ax.set_xlabel("Trim")
ax.set_title("Selling Price of Ford F150 by Trim", fontsize=14, fontweight="bold", x=0.5, y=1.03)

with st.expander(label="Code | Bar, selling price of Ford F150 by trim: "):
	st.code("""
fig, ax = plt.subplots()
sns.barplot(
	x="trim",
	y="sellingprice",
	data=f150s,
	hue="trim",
	palette="husl",
	legend=False,
	gap=0.2,
	estimator=np.mean,
	errorbar="se",
	capsize=0.1,
	ec="0.0",
	lw=0.2,
	alpha=0.8,
	ax=ax
)
plt.xticks(rotation=45)
ax.set_ylabel("Selling Price")
ax.set_xlabel("Trim")
ax.set_title("Selling Price of Ford F150 by Trim", fontsize=14, fontweight="bold", x=0.5, y=1.03)
	""")
with st.container(height=500):
	st.pyplot(fig=fig)



## header -2
st.markdown("""
## 1.9. Now, let's continue with conditon and selling price relationship, but specifically for the selected trims of f150s model this time
""", unsafe_allow_html=True)

## graph-8 jointplot: selling price of Ford F150 by Condition
trim_list = ['XL', 'XLT', 'Platinum', 'SVT Raptor']
g = sns.jointplot(
	x="condition",
	y="sellingprice",
	data=f150s.query("trim in @trim_list"),
	hue="trim",
	palette="husl",
	legend="brief",
	kind="scatter",
	marginal_ticks=True,
	ratio=5,
	space=0.5,
	dropna=True,
	joint_kws={"marker":"d"},
	marginal_kws={"bw_adjust":0.5}
)
g.fig.suptitle("Selling Price Distribution of Ford F150 by Condition", fontsize=14, fontweight="bold", x=0.5, y=1.03)

with st.expander(label="Code | Jointplot, selling price of Ford F150 by condition: "):
	st.code("""
trim_list = ['XL', 'XLT', 'Platinum', 'SVT Raptor']
g = sns.jointplot(
	x="condition",
	y="sellingprice",
	data=f150s.query("trim in @trim_list"),
	hue="trim",
	palette="husl",
	legend="brief",
	kind="scatter",
	marginal_ticks=True,
	ratio=5,
	space=0.5,
	dropna=True,
	joint_kws={"marker":"d"},
	marginal_kws={"bw_adjust":0.5}
)
g.fig.suptitle("Selling Price Distribution of Ford F150 by Condition", fontsize=14, fontweight="bold", x=0.5, y=1.03)
	""")
with st.container(height=500):
	st.pyplot(fig=g)

# header -3
st.markdown("""
### 1.9.1. Findings
* XL and XLT trims are the cheapest trims for a respectively higher condition levels.
""", unsafe_allow_html=True)



## header -2
st.markdown("""
## 1.10. Let's continue with the relationship between selling price - mmr difference (from now on, 'price difference') and color, to see which colors receive less amount than predicted or vice versa
""", unsafe_allow_html=True)

## graph-10 bar, price difference of FORD F150 by color
f150s.loc[:, "PriceDifference"] = f150s.loc[:, "sellingprice"] - f150s.loc[:, "mmr"]

fig, ax = plt.subplots()
sns.barplot(
	x="color",
	y="PriceDifference",
	data=f150s,
	hue="color",
	legend=False,

	gap=0.2,

	estimator=np.mean,
	errorbar=None,
	ax=ax
)
plt.xticks(rotation=45)
ax.tick_params(bottom=True, direction="out")
ax.set_ylabel("Price Difference")
ax.set_xlabel("Color")
ax.set_title("Price Difference of Ford F150 by Color", fontsize=14, fontweight="bold", x=0.5, y=1.03)

with st.expander(label="Code | Bar, price difference of FORD F150 by color: "):
	st.code("""
f150s.loc[:, "PriceDifference"] = f150s.loc[:, "sellingprice"] - f150s.loc[:, "mmr"]

fig, ax = plt.subplots()
sns.barplot(
	x="color",
	y="PriceDifference",
	data=f150s,
	hue="color",
	legend=False,

	gap=0.2,

	estimator=np.mean,
	errorbar=None,
	ax=ax
)
plt.xticks(rotation=45)
ax.tick_params(bottom=True, direction="out")
ax.set_ylabel("Price Difference")
ax.set_xlabel("Color")
ax.set_title("Price Difference of Ford F150 by Color", fontsize=14, fontweight="bold", x=0.5, y=1.03)
	""")
with st.container(height=500):
	st.pyplot(fig=fig)

# header -3
st.markdown("""
### 1.10.1. Findings
* charcoal, silver, and off-white colors are the colors that provide the most profit.
* Orange is the single color that renders buyer disadvantageous. 
""", unsafe_allow_html=True)



## header -2
st.markdown("""
## 1.11. Let's focus on the selected trims that are not orange and the year of which are between 2009 and 2014
""", unsafe_allow_html=True)

## graph-10 heatmap, Seeling prices of Ford F150 by trim and condition (color!='orange, 2009 < year < 2014')
f150s_pivot = (f150s
	.query("trim in @trim_list and color != 'orange' and 2009 < year < 2014")
	.pivot_table(index="trim", columns="condition_bins", values="sellingprice", aggfunc="mean", observed=True))

fig, ax = plt.subplots()
sns.heatmap(
	data=f150s_pivot,
	mask = f150s_pivot > 35000,
	square=False,
	center=20000,
	annot=True,
	fmt=".0f",
	cbar=True,
	cbar_kws={"ticks":(10000, 20000, 34000)},
	cmap="coolwarm",
	shading="flat",
	ax=ax
)
ax.set_ylabel("Trim")
ax.set_xlabel("Condition Bins")
ax.set_title("Selling Prices of Ford F150 by Trim and Condition (color != orange - 2009 < year < 2014", fontsize=14, fontweight="bold", x=0.5, y=1.03)

with st.expander(label="Code | Heatmap, Seeling prices of Ford F150 by trim and condition (color!='orange, 2009 < year < 2014'): "):
	st.code("""
f150s_pivot = (f150s
	.query("trim in @trim_list and color != 'orange' and 2009 < year < 2014")
	.pivot_table(index="trim", columns="condition_bins", values="sellingprice", aggfunc="mean", observed=True))

fig, ax = plt.subplots()
sns.heatmap(
	data=f150s_pivot,
	mask = f150s_pivot > 35000,
	square=False,
	center=20000,
	annot=True,
	fmt=".0f",
	cbar=True,
	cbar_kws={"ticks":(10000, 20000, 34000)},
	cmap="coolwarm",
	shading="flat",
	ax=ax
)
ax.set_ylabel("Trim")
ax.set_xlabel("Condition Bins")
ax.set_title("Selling Prices of Ford F150 by Trim and Condition (color != orange - 2009 < year < 2014", fontsize=14, fontweight="bold", x=0.5, y=1.03)
	""")
with st.container(height=500):
	st.pyplot(fig=fig)

# header -3
st.markdown("""
### 1.11.1. Findings
* XL is the most expensive trim of F150s.
* XLT is a middle-way trim.
""", unsafe_allow_html=True)



## header -2
st.markdown("""
## 1.12. Finally, let's focus on the states to buy the trim XLT with a condition level of 3.5 and above, to see the price difference by state
""", unsafe_allow_html=True)

## graph-11 bar, non-orange Ford F150 XLT of year between 2009 and 2014 price difference and count by state
f150s_reduced = (f150s
	.query("trim == 'XLT' and color != 'orange' and 2009 < year < 2014 and 3.5 <= condition")
	.sort_values(by="state")
	.assign(Count= lambda x:
			x.groupby("state")
				["state"].transform(lambda x: x.count()))
				)

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
a1 = ax[0]
a2 = ax[1]

sns.barplot(
	x="state",
	y="PriceDifference",
	data=f150s_reduced,
	ax=a1,
	hue="state",
	legend=False,
	gap=0.2,
	estimator=np.mean,
	errorbar=None,	
)

sns.barplot(
	x="state",
	y="Count",
	data=f150s_reduced,
	ax=a2,
	hue="state",
	legend=False,
	gap=0.2,
	errorbar=None,
)
a1.set_ylabel("Price Difference")
a2.set_xlabel("State")
fig.suptitle("Non-Orange Ford F150 XLT of Year Between 2009 and 2014 Price Difference and Count by State", fontsize=14, fontweight="bold", x=0.5, y=1)

with st.expander(label="Code | Bar, non-orange Ford F150 XLT of year between 2009 and 2014 price difference and count by state: "):
	st.code("""
f150s_reduced = (f150s
	.query("trim == 'XLT' and color != 'orange' and 2009 < year < 2014 and 3.5 <= condition")
	.sort_values(by="state")
	.assign(Count= lambda x:
			x.groupby("state")
				["state"].transform(lambda x: x.count()))
				)

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
a1 = ax[0]
a2 = ax[1]

sns.barplot(
	x="state",
	y="PriceDifference",
	data=f150s_reduced,
	ax=a1,
	hue="state",
	legend=False,
	gap=0.2,
	estimator=np.mean,
	errorbar=None,	
)

sns.barplot(
	x="state",
	y="Count",
	data=f150s_reduced,
	ax=a2,
	hue="state",
	legend=False,
	gap=0.2,
	errorbar=None,
)
a1.set_ylabel("Price Difference")
a2.set_xlabel("State")
fig.suptitle("Non-Orange Ford F150 XLT of Year Between 2009 and 2014 Price Difference and Count by State", fontsize=14, fontweight="bold", x=0.5, y=1)
	""")
with st.container(height=500):
	st.pyplot(fig=fig)

## header -3
st.markdown("""
### 1.12.1. Findings
* Mission is to find the state where the price difference is considerably low and good amount of XLTs are sold.
* The states "ab" and "ut" conforms to the specification.
""", unsafe_allow_html=True)

warnings.warn = original_warn