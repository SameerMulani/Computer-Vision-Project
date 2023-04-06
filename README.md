# Computer-Vision-Project
Abandoned Baggage detection at airport

Used yolov4 object detection model to identify person and baggages. After detecting baggages, our objective is to determine whether the baggage is abandoned or not.


APPROACH
![Untitled (1)](https://user-images.githubusercontent.com/88852494/230452812-b4e0fb76-7d1b-4837-98bb-56f2d47678d4.jpg)

Firstly, finding the euclidean distance between every person and every baggage. (Euclidean distance between the centroids of person and baggage)

-If the euclidean distance is less than the euclidean threshold value set by us, then we move on to calculate the intersection of union(IOU) between the bounding boxes of the person and 
baggage being considered.
  -If the IOU is greater than the IOU threshold set by us, then it is classified as 'NOT ABANDONED' baggage, else it is suspected to be abandoned.

The remaining baggages are then classified as 'ABONDONED' baggage.


