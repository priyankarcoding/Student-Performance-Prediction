import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title='Student Performance Dashboard', page_icon='🎓', layout='wide')

@st.cache_resource
def load_models():
    path = Path(__file__).parent / 'student_model.pkl'
    if not path.exists():
        path = Path(__file__).parent / 'student_model(1).pkl'
    return joblib.load(path)

try:
    m = load_models()
except Exception as e:
    st.error(f'Model loading failed: {e}. Install the versions in requirements.txt.')
    st.stop()
df = m['training_data']
metrics = m['metrics']
st.title('🎓 Student Performance & Academic Risk Dashboard')
st.caption('Student Performance Prediction – SEC C')
page = st.sidebar.radio('Navigation', ['Overview', 'Preprocessing & Analysis', 'Model Performance', 'Single Student', 'Upload New Dataset', 'Risk & Segmentation', 'Feature Importance'])

def predict(data):
    X = data[list(m['features'])].copy()
    for c in m['num']:
        X[c] = pd.to_numeric(X[c], errors='coerce')
        X.loc[~X[c].between(0, 24 if c == 'study_hours' else 100), c] = np.nan
    p = m['logistic'].predict_proba(X)[:, 1]
    res = data[['student_id']].copy()
    res['Predicted Marks'] = m['linear'].predict(X).clip(0, 100).round(2)
    res['Pass Probability (%)'] = (p * 100).round(2)
    res['Fail Probability (%)'] = ((1-p)*100).round(2)
    res['Result'] = np.where(p >= .5, 'Pass', 'Fail')
    res['Risk'] = np.where(1-p < .3, 'Low', np.where(1-p < .6, 'Medium', 'High'))
    res['KNN Result'] = np.where(m['knn_classifier'].predict(X) == 1, 'Pass', 'Fail')
    clusters = m['kmeans'].predict(m['cluster_pre'].transform(X[m['num']]))
    res['Segment'] = pd.Series(clusters, index=data.index).map(m['segments'])
    _, idx = m['knn'].kneighbors(m['knn_pre'].transform(X))
    res['Similar Students'] = [', '.join(m['reference'].iloc[i]['student_id'].astype(str)) for i in idx]
    return res

if page == 'Overview':
    a,b,c,d = st.columns(4)
    a.metric('Training Students', len(df))
    b.metric('Average Marks', round(df['final_marks'].mean(),2))
    c.metric('Pass Rate', f"{df['pass_fail'].mean()*100:.2f}%")
    d.metric('Test Accuracy', f"{metrics['Accuracy']*100:.2f}%")
    st.dataframe(df)
    st.caption('Original saved dataset; no retraining on uploads.')
    st.plotly_chart(px.histogram(df,x='final_marks',title='Final Marks Distribution'))
    st.plotly_chart(px.pie(df,names=df['pass_fail'].map({0:'Fail',1:'Pass'}),title='Actual Pass/Fail'))

elif page == 'Preprocessing & Analysis':
    st.header('Preprocessing & Analysis')
    st.write('Saved pipeline: median imputation, IQR clipping, scaling and categorical encoding.')
    st.dataframe(df.isna().sum().rename('Missing values in saved data'))
    st.dataframe(df[list(m['num'])+['final_marks']].describe())
    st.plotly_chart(px.imshow(df[list(m['num'])+['final_marks']].corr(),text_auto='.2f',title='Correlation Heatmap'))
    for c in m['num']:
        st.plotly_chart(px.scatter(df,x=c,y='final_marks',title=f'{c} vs Final Marks'))
    st.plotly_chart(px.scatter(df,x='previous_marks',y='final_marks',color='segment',title='Student Groups'))
    st.plotly_chart(px.line(x=list(range(1,len(m['wcss'])+1)),y=m['wcss'],markers=True,title='Elbow Method'))
    st.plotly_chart(px.histogram(m['training_results'],x='risk',color='risk',title='Academic Risk (Original Dataset)'))
    st.caption('Original dataset predictions include training records.')

elif page == 'Model Performance':
    st.header('Model Performance')
    a,b,c,d = st.columns(4)
    a.metric('Accuracy', f"{metrics['Accuracy']*100:.2f}%")
    b.metric('R²', round(metrics['R2'],3))
    c.metric('MAE', round(metrics['MAE'],2))
    d.metric('RMSE', round(metrics['RMSE'],2))
    st.dataframe(pd.DataFrame(metrics.items(),columns=['Metric','Value']))
    st.plotly_chart(px.bar(x=['Logistic Regression','KNN'],y=[metrics['Accuracy']*100,metrics['KNN Accuracy']*100],labels={'x':'Model','y':'Accuracy (%)'},title='Test Accuracy Comparison'))
    st.plotly_chart(px.scatter(x=np.asarray(m['actual']),y=m['predicted'],labels={'x':'Actual Marks','y':'Predicted Marks'},title='Actual vs Predicted'))
    st.plotly_chart(px.imshow(m['confusion'],text_auto=True,x=['Fail','Pass'],y=['Fail','Pass'],labels={'x':'Predicted','y':'Actual'},title='Confusion Matrix'))
    fig=px.line(x=m['fpr'],y=m['tpr'],labels={'x':'False Positive Rate','y':'True Positive Rate'},title=f"ROC Curve (AUC = {m['auc']:.3f})")
    fig.add_shape(type='line',x0=0,y0=0,x1=1,y1=1,line_dash='dash')
    st.plotly_chart(fig)
elif page == 'Feature Importance':
    st.header('Feature Importance')
    importance = m['importance'].copy()
    importance.columns = importance.columns.str.lower().str.strip()
    st.plotly_chart(px.bar(importance,x='importance',y='feature',orientation='h',title='Feature Importance'))

elif page == 'Upload New Dataset':
    file = st.file_uploader('Upload New Student CSV',type='csv')
    if file is not None:
        data = pd.read_csv(file)
        missing = set(['student_id']+list(m['features']))-set(data.columns)
        if missing or data.empty:
            st.error(f'Missing columns: {sorted(missing)}' if missing else 'Dataset is empty')
            st.stop()
        st.dataframe(data)
        try:
            res = predict(data)
        except Exception as e:
            st.error(f'Prediction failed: {e}')
            st.stop()
        a,b,c,d = st.columns(4)
        a.metric('Students',len(res))
        b.metric('Average Predicted Marks',round(res['Predicted Marks'].mean(),2))
        c.metric('Predicted Pass',int((res['Result']=='Pass').sum()))
        d.metric('High Risk',int((res['Risk']=='High').sum()))
        st.dataframe(res)
        st.plotly_chart(px.pie(res,names='Result',title='Pass/Fail Prediction'))
        st.plotly_chart(px.histogram(res,x='Risk',color='Risk',title='Academic Risk',color_discrete_map={'Low':'green','Medium':'orange','High':'red'}))
        st.plotly_chart(px.histogram(res,x='Segment',color='Segment',title='Student Groups'))
        st.plotly_chart(px.histogram(res,x='Predicted Marks',title='Predicted Marks Distribution'))
        st.plotly_chart(px.bar(res,x='student_id',y='Predicted Marks',color='Risk',title='Student Predictions'))
        st.subheader('Top 10 Highest Risk Students')
        st.dataframe(res.nlargest(10,'Fail Probability (%)'))
        st.download_button('Download Results',res.to_csv(index=False),'prediction_results.csv','text/csv')

elif page == 'Risk & Segmentation':
    st.header('Risk & Segmentation')
    results = m['training_results']
    st.plotly_chart(px.histogram(results,x='risk',color='risk',title='Academic Risk',color_discrete_map={'Low':'green','Medium':'orange','High':'red'}))
    st.plotly_chart(px.histogram(results,x='segment',color='segment',title='Performance Groups'))
    st.plotly_chart(px.scatter(df,x='previous_marks',y='final_marks',color='segment',title='K-Means Student Groups'))
    st.dataframe(results.sort_values('fail_probability',ascending=False).head(20))
    st.caption('Original dataset predictions include training records.')

else:
    st.header('Single Student Prediction')
    with st.form('student'):
        sid=st.text_input('Student ID','NEW001')
        attendance=st.slider('Attendance',0,100,75)
        hours=st.slider('Study Hours',0.0,24.0,4.0)
        previous=st.slider('Previous Marks',0,100,60)
        assignment=st.slider('Assignment Marks',0,100,65)
        internal=st.slider('Internal Marks',0,100,60)
        gender=st.selectbox('Gender',['Male','Female'])
        education=st.selectbox('Parental Education',['School','Graduate','Postgraduate'])
        activity=st.selectbox('Extracurricular',['No','Yes'])
        submitted=st.form_submit_button('Predict')
    if submitted:
        data = pd.DataFrame([dict(student_id=sid, attendance=attendance, study_hours=hours, previous_marks=previous, assignment_marks=assignment, internal_marks=internal, gender=gender, parental_education=education, extracurricular=activity)])
        try:
            result = predict(data)
            st.session_state['student_input'] = data
            st.session_state['student_result'] = result
        except Exception as e:
            st.error(f'Prediction failed: {e}')
            st.stop()

    if 'student_result' in st.session_state:
        data = st.session_state['student_input']
        res = st.session_state['student_result']
        r = res.iloc[0]
        st.subheader(f"Student Analysis: {r['student_id']}")
        a,b,c,d = st.columns(4)
        a.metric('Predicted Marks', r['Predicted Marks'])
        b.metric('Result', r['Result'])
        c.metric('Risk', r['Risk'])
        d.metric('Student Group', r['Segment'])
        st.dataframe(res)
        a,b = st.columns(2)
        a.plotly_chart(go.Figure(go.Indicator(mode='gauge+number', value=r['Predicted Marks'], title={'text':'Predicted Marks'}, gauge={'axis':{'range':[0,100]},'bar':{'color':'royalblue'},'steps':[{'range':[0,40],'color':'#ef4444'},{'range':[40,70],'color':'#eab308'},{'range':[70,100],'color':'#22c55e'}],'threshold':{'value':40,'line':{'color':'black','width':3}}})))
        fig = go.Figure(go.Pie(labels=['Pass','Fail'],values=[r['Pass Probability (%)'],r['Fail Probability (%)']],hole=.65,marker_colors=['#16a34a','#ef4444']))
        fig.update_layout(title='Pass / Fail Probability',annotations=[dict(text=f"{r['Pass Probability (%)']:.1f}%<br>Pass",x=.5,y=.5,showarrow=False,font_size=22)])
        b.plotly_chart(fig)
        cols = ['attendance','previous_marks','assignment_marks','internal_marks']
        comparison = pd.DataFrame({'Factor':cols,'Student':data[cols].iloc[0].values,'Dataset Average':df[cols].mean().values})
        st.plotly_chart(px.bar(comparison,x='Factor',y=['Student','Dataset Average'],barmode='group',title='Academic Performance Comparison'))
        a,b = st.columns(2)
        radar = go.Figure()
        for name in ['Student','Dataset Average']:
            radar.add_trace(go.Scatterpolar(r=comparison[name],theta=comparison['Factor'],fill='toself',name=name))
        radar.update_layout(title='Performance Radar',polar=dict(radialaxis=dict(range=[0,100])))
        a.plotly_chart(radar)
        b.plotly_chart(go.Figure(go.Indicator(mode='gauge+number',value=r['Fail Probability (%)'],number={'suffix':'%'},title={'text':'Academic Risk'},gauge={'axis':{'range':[0,100]},'steps':[{'range':[0,30],'color':'lightgreen'},{'range':[30,60],'color':'orange'},{'range':[60,100],'color':'lightcoral'}]})))
        st.plotly_chart(px.bar(x=['Student','Dataset Average'],y=[data['study_hours'].iloc[0],df['study_hours'].mean()],labels={'x':'Category','y':'Study Hours'},title='Study Hours Comparison'))
        st.plotly_chart(px.bar(x=['Previous','Assignment','Internal','Predicted Final'],y=[data['previous_marks'].iloc[0],data['assignment_marks'].iloc[0],data['internal_marks'].iloc[0],r['Predicted Marks']],labels={'x':'Assessment','y':'Marks'},title='Individual Marks Analysis'))
        st.subheader('Academic Recommendations')
        if r['Risk'] == 'High':
            st.warning('Additional academic support is recommended.')
        elif r['Risk'] == 'Medium':
            st.warning('Regular academic monitoring is recommended.')
        else:
            st.success('Maintain consistent academic performance.')
        if data['attendance'].iloc[0] < 75:
            st.write('Improve class attendance.')
        if data['study_hours'].iloc[0] < 3:
            st.write('Consider increasing study hours.')
        st.caption('Predictions are estimates. Recommendations use simple thresholds.')
        st.download_button('Download Student Report',res.to_csv(index=False),'student_report.csv','text/csv')
