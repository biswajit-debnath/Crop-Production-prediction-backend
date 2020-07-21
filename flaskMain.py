import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree
from ipykernel import kernelapp as app
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pyrebase
import math 
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS



app = Flask(__name__)
CORS(app)



config = {
    "apiKey": "AIzaSyCJxL4rNiXVy9PNQbgpjcBtpmaFt3i_dvU",
    "authDomain": "notional-access-711.firebaseapp.com",
    "databaseURL": "https://notional-access-711.firebaseio.com",
    "projectId": "notional-access-711",
    "storageBucket": "notional-access-711.appspot.com",
    "messagingSenderId": "471371191646",
    "appId": "1:471371191646:web:039a421983d13564fe6bcf",
    "measurementId": "G-F2ZSVFCHPB"
}

firebase = pyrebase.initialize_app(config)

storage = firebase.storage()







def prediction(State, year,df):
    
	state_df=df[df.State_Name == State]
	if(state_df.empty):
		return 0

	#grouping area and production for each year by average
	data=state_df.groupby(['Crop_Year'])['Area','Production'].mean()
	data=data.reset_index(level=0, inplace=False)
	

	#calulation cpi(crop production index)
	#cpi=production/area
	data['CPI']=data['Production']
	



	#analysis of variations in production
	if(State == "Assam"):
		x_axis=data.Crop_Year

		y1_axis=data.Production

		fig = plt.figure(figsize=(16,8))
		plt.plot(x_axis,y1_axis)
		ax = fig.add_subplot(111)
		ax.title.set_fontsize(30)
		ax.xaxis.label.set_fontsize(20)
		ax.yaxis.label.set_fontsize(20)
		plt.title(" PRODUCTION ")
		plt.legend("PRODUCTION")
		plt.savefig('fig.png')
		plt.clf()
		storage.child("images/fig.png").put("fig.png")
		url1=storage.child('images/fig.png').get_url(None)




	#analysis of variation in area
	if(State == "Assam"):
		x_axis=data.Crop_Year
		y_axis=data.Area

		y1_axis=data.Production
		fig = plt.figure(figsize=(16,8))
		plt.plot(x_axis,y_axis)
		ax = fig.add_subplot(111)
		ax.title.set_fontsize(30)
		ax.xaxis.label.set_fontsize(20)
		ax.yaxis.label.set_fontsize(20)
		plt.title("Area ")
		plt.legend(["Area"])
		plt.savefig('fig1.png')
		plt.clf()
		storage.child("images/fig1.png").put("fig1.png")
		url1=storage.child('images/fig1.png').get_url(None)




	#details about mean,std deviation,min,maxetc
	
	data = data[np.isfinite(data['CPI'])]



	x=data.iloc[:,0:1].values
	y=data.iloc[:,3].values
	regressor=RandomForestRegressor(n_estimators=8,random_state=0,n_jobs=1,verbose=35)
	regressor.fit(x,y)




	#predicting for the test values
	if(State == "Assam"):
		y_pred=regressor.predict(x)
		x_grid=np.arange(min(x),max(x),0.001)
		x_grid=x_grid.reshape(len(x_grid),1)
		plt.scatter(x,y,color='r')
		plt.plot(x_grid,regressor.predict(x_grid),color='b')
		plt.savefig('fig2.png')
		plt.clf()
		storage.child("images/fig2.png").put("fig2.png")
		url1=storage.child('images/fig2.png').get_url(None)
		
		






	#actual and predicted values
	if(State == "Assam"):
		dm = pd.DataFrame({'Actual': y, 'Predicted': y_pred}).reset_index()
		x_axis=dm.index
		y_axis=dm.Actual
		y1_axis=dm.Predicted

		fig = plt.figure(figsize=(15,15))

		plt.plot(x_axis,y_axis)
		plt.plot(x_axis,y1_axis)
		plt.title("Actual vs Predicted")
		ax = fig.add_subplot(111)
		ax.title.set_fontsize(30)
		ax.xaxis.label.set_fontsize(20)
		ax.yaxis.label.set_fontsize(20)
		plt.legend(["actual ","predicted"],fontsize=30)
		plt.savefig('fig3.png')
		plt.clf()
		storage.child("images/fig3.png").put("fig3.png")
		url2=storage.child('images/fig3.png').get_url(None)
		
		



	newx=[[year]]
	pred_new=regressor.predict(newx)
	if(State == "Assam"):
		return pred_new,url1,url2

	return pred_new









def fn(Crop):
	df=pd.read_csv("apy.csv",encoding = "ISO-8859-1")
	result=[]
	sorted_result=np.array([])
	url1=''
	url2=''
	states = df.State_Name.unique()

	selected_crops=['Rice','Banana','Groundnut','Sunflower']
	
	df.dropna(subset=['Production'], how='all', inplace=True)
	state_df=df[df.Crop == Crop]
	for state in states:
	    output=prediction(state,2016,state_df)
	    result.append({"state":state, "Production":0 if output == 0 else math.floor(output[0])})
	    if(state == "Assam" and output != 0):
	    	url1=output[1];
	    	url2=output[2];

	
	
	sorted_result=sorted(result, key = lambda i: i['Production'],reverse=True)
	# print(result)
	#json.dumps(result)
	
	return {"data":sorted_result, "url1":url1, "url2":url2}








@app.route('/', methods=['POST'])
def basic():
	data=request.form['value']
	output=fn(data);
	print("Data",data)
	return jsonify(output)



if __name__ == '__main__':
	app.run(host= '0.0.0.0',port=8080)