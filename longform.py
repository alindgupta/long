#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi']=300

infilename = 'data/2Lcohort.csv' # File to read from
outfilename = 'data/long2L.csv' # File name to write

# Variables in infile
idvar = 'PatientID' # Id variable
t0 = 't0' # time zero date
eventdate = 'DateOfDeath' # Date of death
maxvisit = 'maxvisit' # Date of last visit for censoring purposes
event_tte = 'OS' # time to event variable
event_flag = 'EVENT' # event indicator 

# Long formatting
NUM_DAYS = 30 # Number of days within 1 "period"
MAX_DATE = pd.to_datetime("2020-10-01") # maximum allowed date (e.g. administrative cutoff)
MAX_DAYS = 360 * 6 # maximum follow-up cutoff (here 6 years)
num_days_censor = 30 # number of days after last record to censor (here 30 days)

if MAX_DAYS % NUM_DAYS != 0:
    raise('Please ensure maximum allowed days of follow-up is a multiple of period')

df = pd.read_csv(infilename)
df = df[[idvar, t0, eventdate, maxvisit]].drop_duplicates()
df = df.astype({col: 'datetime64' for col in (t0, eventdate, maxvisit)})

# Define start and end dates
# Start date is t0 
# End date is minimum of MAX_DATE or MAX_DAYS after t0
df['MAX_DATE'] = MAX_DATE
df['MAX_DAYS'] = df[t0] + pd.Timedelta(360*6, unit='d')
df['end_date'] = df[['MAX_DATE', 'MAX_DAYS']].min(axis=1)
df = df.drop(['MAX_DATE', 'MAX_DAYS'], axis=1)

# Plot end dates with respect to maxvisit
plt.scatter(df['end_date'], df[maxvisit], s=2)
plt.xlabel('Trial end date')
plt.ylabel('Max EMR date')
plt.show()

# Patients are censored if they don't have event date
# using num_days_censor days after last visit date
# if last visit date is num_days_censor days after the trial end date for them
# Patients are not censored if they have a date of death that is 
# later than end date
df['censor'] = (
    (df[maxvisit] < (df['end_date'] - pd.Timedelta(num_days_censor, unit='d'))) &
    (df[eventdate].isnull()))

# Some plotting
x = df.copy()
x = x.sort_values([t0])
cex = 4
plt.plot_date(x[idvar], x[eventdate], markersize=cex*0.5,
              zorder=2,
              c='#e24a33',
              label='Event date', alpha=1)
plt.xlabel('Patients')
plt.ylabel('Time')
plt.xticks([])
plt.title('Overview of trial')
plt.scatter(x[idvar], x[t0], color='#348abd', s=cex, zorder=4, marker='^', label='Time zero')
plt.scatter(x[idvar], x['end_date'], color='goldenrod', s=cex, zorder=1, marker='^', label='End of follow-up')
plt.scatter(x.loc[x.censor, idvar], x.loc[x.censor, 'end_date'], 
            color='b', s=cex*0.1, zorder=4, marker='.', label='Censored')

plt.legend(loc='lower right', markerscale=3, prop={'size': 9})
plt.show()

# Person-time should only go upto event date (first event date) if 
# event occurs before trial end
df['end_date'] = df[[eventdate, 'end_date']].min(axis=1)

# Convert to long between t0 and end date per patient
df = pd.concat(
    [pd.DataFrame(
        {idvar : row[idvar],
         't': pd.date_range(row[t0],
                            row['end_date'] + pd.Timedelta(NUM_DAYS, unit='d'), # to encompass the end date
                            freq=f'{NUM_DAYS}D'), # by 30 days
         t0: row[t0],
         'end_date': row['end_date'],
         maxvisit: row[maxvisit],
         'censor': row.censor,
         eventdate: row[eventdate]})
     for i, row in df.iterrows()], ignore_index=True)
df = df.drop_duplicates()
df['month'] = df.groupby(idvar).cumcount()

# Events and censoring
isnaevent = (df[eventdate].isnull())
censored = df['censor']
df[['EVENT', 'CENS']] = 0

# Event - have event, t>event date
# Censoring is irrelevant because the person must have been
# followed up until their death
df.loc[~isnaevent & (df.t >= df[eventdate]), 'EVENT'] = 1

# Censored at num_days_censor days after last visit
# See https://doi.org/10.1093/aje/kwx281 "When to censor"
df.loc[isnaevent & censored & (df.t > (df[maxvisit] + pd.Timedelta(num_days_censor, unit='d'))), 'EVENT'] = np.nan
df.loc[isnaevent & censored & (df.t > (df[maxvisit] + pd.Timedelta(num_days_censor, unit='d'))), 'CENS'] = 1

# Create censoring date for censored patients num_days_censor after last visit
df['censor_date'] = df[maxvisit] + pd.Timedelta(num_days_censor, unit='d')
df.loc[~censored, 'censor_date'] = np.nan

# Truncate rows after the time point when censoring first occurs
df['censor_t'] = (df['t'] - df['censor_date']).dt.days
df['censor_t'] = df.groupby(idvar)['censor_t'].transform(lambda i: (i[i >= 0]).min())
df['censor_t'] = df['censor_date'] + pd.to_timedelta(df['censor_t'], unit='d')
df = df.loc[(df['censor_t'].isnull()) | (df['t'] <= df['censor_t'])]

# Plot to ensure person-period conversion matches time to event data
x = df.copy()
x['tt'] = x.groupby(idvar)['month'].transform('max')

y = pd.read_csv(infilename)
y = y.rename(columns={event_flag: 'BL', event_tte: 'AVAL'})

x = x.merge(y, how='left', on=idvar)
x = x[[idvar, 'tt', 'AVAL', 'CENS']].drop_duplicates()

plt.scatter(x.AVAL[x.CENS==1], x.tt[x.CENS==1], zorder=2, marker='o', label='Censored', s=3)
plt.scatter(x.AVAL[x.CENS==0], x.tt[x.CENS==0], zorder=1, marker='o', label='Not censored', s=3)

plt.title('Time-to-event vs long-format')
plt.xlabel('OS (time-to-event)')
plt.ylabel('Number of months (long format)')
ident = [0.0, 60]
plt.plot(ident, ident, color='black', ls='--')
plt.legend(loc='lower right', markerscale=3)
plt.show()

df.to_csv(outfilename, index=False)
