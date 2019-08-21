var pic = null;  //存放图片

function pictureUpload(){
    //alert("yes")
    $('#inputPic').click();   //模拟input按钮的点击效果 
}

/**
 * 缩略图预览
 * @param file
 * @param container
 */
var preview = function(file, container){
    //缩略图类定义
    var Picture  = function(file, container){
        var height = 0,
            width  = 0,
            ext    = '',
            size   = 0,
            name   = '',
            path   =  '';
        var self   = this;
        if(file){
            name = file.value;
            if(window.navigator.userAgent.indexOf("MSIE")>=1){
                file.select();
                path = document.selection.createRange().text;
            }else {
                if(file.files){
                    // path =  file.files.item(0).getAsDataURL(); // firefox7.0之后该方法弃用了，用下面那个
                    path = window.URL.createObjectURL(file.files[0]);
                }else{
                    path = file.value;
                }
            }
        }else{
            throw '无效的文件';
        }
        ext = name.substr(name.lastIndexOf("."), name.length);
        if(container.tagName.toLowerCase() != 'img'){
            throw '不是一个有效的图片容器';
            container.visibility = 'hidden';
        }
   
        pic = $('#inputPic')[0].files[0];
        container.src = path;
        container.alt = name;
        container.style.visibility = 'visible';
        height = container.height;
        width  = container.width;
        size   = container.fileSize;
        this.get = function(name){
            return self[name];
        };
        this.isValid = function(){
            if(allowExt.indexOf(self.ext) !== -1){
                throw '不允许上传该文件类型';
                return false;
            }
        }
    };

    try{
        var pic2 =  new Picture(file, document.getElementById('' + container));
    }catch(e){
        alert(e);
    }
    
};

//ajax上传图片
function upload(){
	//清空识别结果
	$("#img_driver2").attr('src',"../static/assets/images/camera.jpg");
	$("#recognitionResult").val('');
	var formData = new FormData(); 
	//alert(pic);
	formData.append("file", pic);  //添加图片信息的参数s
	$.ajax({
	    url: 'processingSingle',
	    type: 'POST',
	    async : true,//是否异步请求
	    cache: false, //上传文件不需要缓存
	    data: formData,
	    processData: false, // 告诉jQuery不要去处理发送的数据
	    contentType: false, // 告诉jQuery不要去设置Content-Type请求头
	    success: function (data) {
	    	if(data=="error1")
	    		alert("error1");
	    	else if(data=="error2")
	    		alert("error2");
	    	else if(data=="error3")
	    		alert("error3");
	    	else{
	    		var arr = data.split("|");
	    		$("#img_driver2").attr('src',"../static/upload/"+arr[0]+"/detection1/"+arr[0]+".jpg");
	    		$("#recognitionResult").val(arr[1]);
	    		alert("识别完成");
	    	} 		
	    },
	    error: function (data) {
	        alert("上传失败");
	    }
	})  
}
