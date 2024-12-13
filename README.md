# Web-Mining-and-Analytics
**INTRODUCTION**

Crime rates and patterns are critical metrics for ensuring public safety and optimizing law enforcement efforts. The analysis focuses on identifying crime hotspots, examining temporal trends, visualizing crime distributions, predicting future crime incidents, and generating actionable recommendations for law enforcement and community leaders. 


These efforts aim to empower stakeholders, including law enforcement agencies and policymakers, with actionable insights to enhance resource allocation, improve patrol schedules, and implement targeted interventions effectively.

****SOURCE SYSTEMS** **


The primary source of data for this project is the publicly available Chicago crime dataset. This dataset includes detailed attributes such as: 

**Crime Type:** Categorized into various offenses like theft, assault, and motor vehicle theft. 

**Location Descriptions**: Specific locations where crimes occurred, e.g., streets, parks, or residences. 

**Timestamps**: Date and time of each recorded crime, which facilitate temporal analysis. 

**Geospatial Coordinates**: Latitude and longitude to support spatial clustering and mapping. 

This dataset was obtained from the Chicago Open Data Portal and supplemented by additional publicly available datasets to enrich the analysis.  

**Data Preparation**



**Date Parsing**: Converted the Date column to datetime format and set it as the index for time-series operations. 

**Missing Values**: Dropped rows with missing values in critical columns (Primary Type). 

**Time Feature**s: Extracted features such as Day, Month, Year, DayOfWeek, and IsWeekend. 

**Categorical Encoding**: Encoded Primary Type and Location Description using categorical encoding. 

**Daily Aggregation**: Aggregated the data into daily crime counts to facilitate time-series analysis. 
