IMAGE_SRC_PREFIX = "data:image/png;base64,"

$("#upload-receipt").click(function(evt){
    document.getElementById("input-receipt").click()
})

$("#input-receipt").change(function(evt){
    let image = document.getElementById('show-receipt');
    image.src = URL.createObjectURL(evt.target.files[0]);
    getBase64Image(evt.target.files[0])
    .then(base64Img => {
        console.log(base64Img)
        $.post("/uploadReceipt", {receipt: base64Img }, function(data, status){
            if (data){
                if (data == "fail"){
                    alert("Cannot align the given input. Please select another image!")
                    return
                }
                model = JSON.parse(data)
                console.log(model["alignment"])
                if (model["alignment"]){
                    console.log("get here")
                    alignmentSrc = IMAGE_SRC_PREFIX + model["alignment"]
                    document.getElementById("alignment").src = alignmentSrc
                }
                if (model["alignment_crop"]){
                    alignmentCropSrc = IMAGE_SRC_PREFIX + model["alignment_crop"]
                    document.getElementById("alignment-crop").src = alignmentCropSrc
                }
                if (model["field_detection"]){
                    fieldDetectionSrc = IMAGE_SRC_PREFIX + model["field_detection"]
                    document.getElementById("field_detection").src = fieldDetectionSrc
                }
                if (model["ocr"] && model["ocr"] != {}){
                    ocrAttr = model["ocr"]
                    console.log(ocrAttr)
                    if (ocrAttr["market_name"]){
                        marketNameSrc = IMAGE_SRC_PREFIX + ocrAttr["market_name"]["image"]
                        marketNameText = ocrAttr["market_name"]["text"]
                        document.getElementById("market_name").src = marketNameSrc
                    }
                    if (ocrAttr["date"]){
                        dateSrc = IMAGE_SRC_PREFIX + ocrAttr["date"]["image"]
                        dateText = ocrAttr["date"]["text"]
                        document.getElementById("date").src = dateSrc
                    }
                    if (ocrAttr["bill_code"]){
                        billCodeSrc = IMAGE_SRC_PREFIX + ocrAttr["bill_code"]["image"]
                        billCodeText = ocrAttr["bill_code"]["text"]
                        document.getElementById("bill_code").src = billCodeSrc
                    }
                    productAttrs = ocrAttr["product_attributes"]
                    if (Array.isArray(productAttrs)){
                        productAttrs.forEach(function(item){
                            imgSrc = IMAGE_SRC_PREFIX + item["image"]
                            $("#product-attr-board").append(`
                            <div class="col-md-6">
                                <div class="col-md-12">
                                    <img id="alignment" width="300" height="80" src="${imgSrc}" style="border-radius: 10px; margin-top: 10px;"/>
                                </div>
                                <div class="col-md-12">
                                    <code class="field-name begin">SKU:&nbsp;</code><label class="field-name">${item["sku"]}</label><br/>
                                    <code class="field-name begin">product name:&nbsp;</code><label class="field-name">${item["product_name"]}</label><br/>
                                    <code class="field-name begin">quantity:&nbsp;</code><label class="field-name">${item["quantity"]}</label><br/>
                                    <code class="field-name begin">ppu:&nbsp;</code><label class="field-name">${item["ppu"]}</label><br/>
                                    <code class="field-name begin">original price:&nbsp;</code><label class="field-name">${item["original_price"]}</label><br/>
                                    <code class="field-name begin">total price:&nbsp;</code><label class="field-name">${item["total_price"]}</label><br/>
                                </div>
                            </div>`)
                        })
                    }
                }
            }
        });
    })
    .catch(error => {
        console.error(error)
    })
    
    
})
