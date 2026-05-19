# A5. Gauteng School Calendar — Manual

## Source
DBE annual school calendars (PDF, one per year):
  https://www.education.gov.za/Curriculum/SchoolCalendar.aspx
Gauteng-specific opening days are usually identical to national.

## Approximate pattern (verify per year)
  Term 1: mid-Jan        -> late Mar
  Term 2: early Apr      -> mid-Jun
  Term 3: mid-Jul        -> late Sep
  Term 4: early Oct      -> early Dec
  Exam periods: last ~3 weeks of each term.

## Why
Term/exam transitions shift paediatric and adolescent presentations
(injuries, asthma flares, exam-stress mental health). Already-encoded
public holidays (is_public_holiday) miss this layer.

## Workflow
Fill rows for every date in the dataset range. The join script will
treat missing dates as is_school_term=0.
