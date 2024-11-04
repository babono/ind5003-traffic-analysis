function captureTrafficImagesOnCausewayAndSecondLink() {
    const trafficImagesUrl = 'http://datamall2.mytransport.sg/ltaodataservice/Traffic-Imagesv2';
    const headers = { 'AccountKey': 'AO4qMbK3S7CWKSlplQZqlA==' };
    const folderId = '1J_fiRwKe0agb8EhTiWPiiLwMrrq-mzzT';
    const allowedCameraIds = ['2701', '2702', '4703', '4713'];
  
    try {
      // Fetch image metadata
      const options = {
        'headers': headers
      };
      const response = UrlFetchApp.fetch(trafficImagesUrl, options);
      const json = JSON.parse(response.getContentText());
      const folder = DriveApp.getFolderById(folderId);
  
      // Retrieve images from allowed camera IDs and store them in Google Drive
      json.value.forEach(function (item) {
        if (allowedCameraIds.includes(item.CameraID)) {
          const imgLink = item.ImageLink;
          const imgResponse = UrlFetchApp.fetch(imgLink, options);
          const singaporeTime = new Date();
          // Convert to Singapore local time (UTC+8)
          const timestamp = Utilities.formatDate(singaporeTime, 'Asia/Singapore', 'yyyy-MM-dd_HH-mm-ss');
          const filename = `${item.CameraID}_${timestamp}.jpg`;
          const imgBlob = imgResponse.getBlob().setName(filename);
          folder.createFile(imgBlob);
          Logger.log(filename + ' saved to Google Drive.');
        }
      });
    } catch (error) {
      Logger.log('Error occurred: ' + error.message);
    }
  }