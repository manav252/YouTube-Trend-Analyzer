# Power BI DAX Measures

```DAX
Total Videos = COUNTROWS('youtube_cleaned')

Total Views = SUM('youtube_cleaned'[views])

Total Likes = SUM('youtube_cleaned'[likes])

Total Comments = SUM('youtube_cleaned'[comment_count])

Average Views = AVERAGE('youtube_cleaned'[views])

Engagement Rate =
DIVIDE(
    SUM('youtube_cleaned'[likes]) + SUM('youtube_cleaned'[comment_count]),
    SUM('youtube_cleaned'[views])
)

Like Ratio =
DIVIDE(
    SUM('youtube_cleaned'[likes]),
    SUM('youtube_cleaned'[views])
)

Comment Ratio =
DIVIDE(
    SUM('youtube_cleaned'[comment_count]),
    SUM('youtube_cleaned'[views])
)

High Engagement Videos =
CALCULATE(
    COUNTROWS('youtube_cleaned'),
    'youtube_cleaned'[high_engagement] = 1
)

Unique Channels = DISTINCTCOUNT('youtube_cleaned'[channel_title])
```
