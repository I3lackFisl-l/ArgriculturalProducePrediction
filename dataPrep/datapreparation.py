class DataPreparation:
  def castType(self, df):
    cols = df.select_dtypes(exclude=['float','int']).columns.to_list()
    df[cols] = df[cols].astype('category')
    df[['month_no','year_no']] = df[['month_no','year_no']].astype('category')
    return df
  
  def fillMeanMedian(self, df):
    theDataMedian = df.median(numeric_only=True)
    theDataMode = df.mode().iloc[0]
    modeNmedian=theDataMode
    modeNmedian[theDataMedian.index.values.tolist()]=theDataMedian.values.tolist()
    df.fillna(modeNmedian, inplace=True)
    return df

  def IqrOutlier(self, df):
    numerical_cols = df.select_dtypes(include=['float','int']).columns.to_list()
    numerical_cols
    Q1 = df[numerical_cols].quantile(0.25)
    Q3 = df[numerical_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit=Q1 - 1.5 * IQR
    upper_limit=Q3 + 1.5 * IQR

    # print(lower_limit,upper_limit)

    for col in numerical_cols:
      df[col] = np.where(df[col] <lower_limit[col], lower_limit[col], df[col])
      df[col] = np.where(df[col] >upper_limit[col], upper_limit[col], df[col])
    return df

  def __getstate__(self):
    attributes = self.__dict__.copy()
    return attributes
dataPrep = DataPreparation()