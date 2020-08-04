## Visualization of Crime
Visualization of crime statistics for Chicago Neighborhoods, including a neighborhood crime heatmap and a 3D visualization of multiple regression.

## Description
This work is part of a larger class project for NorthWestern Data Science Bootcamp. Everything in this repository represents my work on the project. My role on the project included analyzes relationships between socioeconomic influoences of Chicago neighborhood crime. Specifically, I investigated economic, health, and birth factors on the Chicago homicide rate (per 100,0000 individuals; age-adjusted). There are 77 Chicago neighborhoods. Data was gathered from the Chicago Data Portal. I included the group powerpoint slides that were presented in class on July 28, 2020.

I performed several multiple regression analysis on economic, health, and birth influences on neighborhood homicide. I repeated these analyses on the homicide rate in log units. The homicide rate was transformed into log units to fit a normal distribution.



### Histogram of Homicide Rate
#### Note the positive skew.
    final_df['Homicide_rate_per_100k'].hist(bins=20)
![Histogram of Homicide Rate](</Images/Homicide Rate - Histogram (Positive Skew).png>)

    /# log transformation + constant to make homicide variable more normally distributed
    final_df['Homicide_rate_log'] = final_df['Homicide_rate_per_100k'].apply(lambda x: np.log(x+1))
    final_df['Homicide_rate_log'].hist(bins=20)

#### Distribution is more normal after transforming homicide rate into log units.
![Histogram of After Log Transformation](</Images/Homicide Rate - Histogram (Log Units).png>)


### Simple Regression Function
#### This function will also include language that interprets the strength, direction, and statistical significance of the relationship.
    /# Function for Simple Regression
    /# Features: plots line, provides stats, describes relationship
    def regression(x, y, x_label, y_label):
        title = f'{x_label} predicting {y_label}'
        (slope, intercept, rvalue, pvalue, stderr) = linregress(x, y)
        regress_values = x * slope + intercept
        line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))
        plt.scatter(x, y)
        plt.plot(x,regress_values,"r-")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        title = title.replace('\n', '_').replace(' ', '_')
        plt.savefig(f'output_data/{title}.png')
        plt.show()
        /#plt.close()
        print(line_eq)
        print()
        r = pearsonr(x, y)[0]
        p = pearsonr(x, y)[1]
        print(f"r = {r:.2f}, p = {p:.4f}, r_squared = {rvalue**2:.2f}")
        print()
        if p > 0.05:
            print(f'There is no relationship between {x_label.lower()} and {y_label.lower()}, p > 0.05.')
        else:
            if r > 0:
                explanation = f'An increase in {x_label.lower()} is associated with an increase in {y_label.lower()}.'
                direction = 'positive'
                if r >= 0.6:
                    strength = 'strong'
                elif r >= 0.3:
                    strength = 'moderate'
                else:
                    strength = 'weak'
            else:
                direction = 'negative'
                explanation = f'An increase in {x_label.lower()} is associated with a decrease in {y_label.lower()}.'
                if r <= -0.6:
                    strength = 'strong'
                elif r <= -0.3:
                    strength = 'moderate'
                else:
                    strength = 'weak'       
            print(f'There is a {strength} {direction} relationship between {x_label.lower()} and {y_label.lower()}, p > 0.05.')
            print(explanation)
        print()

### Example Regression Output in Log Units
    /# log units
    /# Regressing neighborhood homicide rate (per 100k) on the Unemployment Rate.
    /# regression(x, y, x_label, y_label)
    regression(final_df['Unemployment'], final_df['Homicide_rate_log'], 'Unemployment (%)', 'Log Units - Homicide Rate')

![Regressing](</Images/Unemployment_(%)_predicting_Log_Units_-_Homicide_Rate.png>)

![Interpretation](</Images/Example Intepretation.PNG>)



### Multiple Regression Function
    /#Features: Provides stats and statistical model test
    def multiple_regression(xs, y): # xs is a dataframe, y is a series; prints output and returns predictions
        xs = sm.add_constant(xs) # adding a constant
        model = sm.OLS(y, xs).fit()
        predictions = model.predict(xs) 
        print_model = model.summary()
        print(print_model)
        return predictions

### 3D Visualizing Multiple Regression Function
    def regression_3d_visualization(dataframe, x1, x2, y, size = (10,10), x1label='X Label', x2label='Y Label', ylabel='Z Label'):
        df = dataframe[[x1, x2, y]].reset_index(drop=True)
        x1r, x2r, yr = x1, x2, y
        if ' ' in x1:
            x1r = x1.replace(' ', '')
            df.rename(columns={x1: x1r}, inplace=True)
        if ' ' in x2:
            x2r = x2.replace(' ', '')
            df.rename(columns={x2: x2r}, inplace=True)
        if ' ' in y:
            yr = y.replace(' ', '')
            df.rename(columns={y: yr}, inplace=True)
        model = smf.ols(formula=f'{yr} ~ {x1r} + {x2r}', data=df)
        results = model.fit()
        results.params

        x_dim, y_dim = np.meshgrid(np.linspace(df[x1r].min(), df[x1r].max(), 100), np.linspace(df[x2r].min(), df[x2r].max(), 100))
        xs = pd.DataFrame({x1r: x_dim.ravel(), x2r: y_dim.ravel()})
        predicted_y = results.predict(exog=xs)
        predicted_y=np.array(predicted_y)

        fig = plt.figure(figsize=size, facecolor='b')
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df[x1r], df[x2r], df[yr], c='red', marker='o', alpha=0.5)
        ax.plot_surface(x_dim, y_dim, predicted_y.reshape(x_dim.shape), color='b', alpha=0.3)
        ax.set_xlabel(x1label)
        ax.set_ylabel(x2label)
        ax.set_zlabel(ylabel)
        plt.show()
 
![3D Regression](</Images/After Log Transformation - Homicide per 100k on Unemployment and TeenBirthRate View.png>)


## Visualizing the Neighborhood Homicide Rate
    import gmaps
    import gmaps.datasets
    import gmaps.geojson_geometries
    import json
    from matplotlib.cm import viridis, plasma
    from matplotlib.colors import to_hex
    import matplotlib.pyplot as plt
    import pandas as pd
    from config import census_key, g_key

    /# Configure gmaps
    gmaps.configure(api_key=g_key)
    ngeo = json.load(open("ncoordinates.json"))

    / #dict_keys(['type', 'crs', 'features'])
    / #print(len(ngeo['features']))
    / #print(ngeo.keys())
    / #print()
    ngeo['features'][0]
    / # Chicago Neighborhoods
    data = pd.read_csv('../data/Neighborhood_Health.csv')
    data_dict = data.filter(['Community Area Name', 'Assault (Homicide)'])
    data_dict = data_dict.set_index('Community Area Name').to_dict()["Assault (Homicide)"]
    rate_max = data[data["Assault (Homicide)"] == data["Assault (Homicide)"].max()]
    rate_max_value = round(rate_max["Assault (Homicide)"],2).values[0]
    rate_min = data[data["Assault (Homicide)"] == data["Assault (Homicide)"].min()]
    rate_min_value = round(rate_min["Assault (Homicide)"],2).values[0]

    #Scale the states values to lie between 0 and 1
    min_nh = min(data_dict.values())
    max_nh = max(data_dict.values())
    nh_range = max_nh - min_nh

    def calculate_color(neighborhood): #Convert the neighborhood value to a color
        normalized_nh = (neighborhood - min_nh) / nh_range # make neighborhood a number between 0 and 1
        inverse_nh = 1.0 - normalized_nh # invert neihborhood so that high value gives dark color
        mpl_color = plasma(inverse_nh) # transform the neighborhood value to a matplotlib color
        gmaps_color = to_hex(mpl_color, keep_alpha=False) # transform from a matplotlib color to a valid CSS color
        return gmaps_color

    colors = []
    for feature in ngeo['features']:
        geo_nh_name = feature['properties']['PRI_NEIGH']
        try:
            nh = data_dict[geo_nh_name]
            color = calculate_color(nh)
        except KeyError:
            # no value for that state: return default color
            color = (0, 0, 0, 0.3)
        colors.append(color)
        
    fig = gmaps.figure()
    nh_layer = gmaps.geojson_layer(
        ngeo,
        fill_color=colors,
        stroke_color=colors,
        fill_opacity=0.8)
    fig.add_layer(nh_layer)
    fig
    
 ![Visualization Neighborhood Homicide Rate](<Images/city_of_chicago_crime_map.png>)
