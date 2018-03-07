#importing dash
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

#import data libraries
import numpy as np
import pandas as pd

# import csv file as data frame
df = pd.read_csv('crops_data_nonzero_protein.csv')
# holds dimensions
shape = df.shape
print ("Original Dimensions : %s" % (shape,))
# crops properties
names = df['Name']
protein = df['Protein']
carbs = df['Carbs']
# selecting features for the model
#feature = df[['Temp_1', 'Temp_2', 'Temp_3', 'Temp_4','Protein']]
feature = df[['Temp_1', 'Temp_2', 'Temp_3', 'Temp_4']]

# import TSNE estimator
from sklearn.manifold import TSNE
# convert data to nparray for TSNE
X = np.array(feature)
# apply dimensionality reduction
X_embedded = TSNE(n_components=2).fit_transform(X)
# prints the new data dimensions
print ("New reduced Dimensions : %s" % (X_embedded.shape,))

# application interface code
app = dash.Dash()

app.layout = html.Div([
    html.H1(children='3D-Scatter plot for crops'),
    dcc.Graph(
        id='crops',
        figure={
            'data': [
                go.Scatter3d(
                    x=X_embedded[:,0],
                    y=X_embedded[:,1],
                    z=carbs,
                    hovertext=names,
                    mode='markers+text',
                    opacity=1,
                    showlegend = True,

                    marker=dict(
                        size=8,
                        color = carbs, # set color to an array/list of desired values
                        colorscale='Rainbow', # choose a colorscale
                        showscale=True,
                        opacity=1,
                        colorbar = dict(
                                title = 'Carbs level'
                        ),
                    )
                )
            ],
            'layout': go.Layout(
                                title = 'Crops temperature VS carb level',
                                autosize=False,
                                width=1200,
                                height=800,
                                margin=go.Margin(
                                    l=50,
                                    r=50,
                                    b=100,
                                    t=100,
                                    pad=4
                                ),
            ),
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)