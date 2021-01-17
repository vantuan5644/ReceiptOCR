function getBase64Image(img) {
    let reader = new FileReader()
    let baseString
    return new Promise((resolve) => {
        reader.onloadend = function () {
            dataURL = reader.result;
            return resolve(dataURL.replace(/^data:image\/(png|jpg|jpeg);base64,/, ""))
        };
        reader.readAsDataURL(img);
    })
  }



  