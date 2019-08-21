var pic = null;  //存放图片
var dataCount=0;//总数据条数
var pageSize=6;//每页显示条数
var timeStamp;//时间戳
var array = [[1, '全局阈值二值化'],
			[2, '高斯模糊1'],
			[3, '高斯模糊2'],
			[4, '随机颜色0'],
			[5, '随机颜色1'],
			[6, '随机颜色2'],
			[7, '随机颜色3'],
			[8, '随机颜色4'],
			[9, '随机颜色5'],
			[10, '随机颜色6'],
			[11, '随机颜色7'],
			[12, '灰度化'],
			[13, '添加随机线条及点0'],
			[14, '添加随机线条及点1'],
			[15, '添加随机线条及点2'],
			[16, '局部阈值二值化'],
			[17, '提取特征值'],
			[18, '随机角度旋转0_全局阈值二值化'],
			[19, '随机角度旋转0_高斯模糊1'],
			[20, '随机角度旋转0_高斯模糊2'],
			[21, '随机角度旋转0_随机颜色0'],
			[22, '随机角度旋转0_随机颜色1'],
			[23, '随机角度旋转0_随机颜色2'],
			[24, '随机角度旋转0_随机颜色3'],
			[25, '随机角度旋转0_随机颜色4'],
			[26, '随机角度旋转0_随机颜色5'],
			[27, '随机角度旋转0_随机颜色6'],
			[28, '随机角度旋转0_随机颜色7'],
			[29, '随机角度旋转0_灰度化'],
			[30, '随机角度旋转0_添加随机线条及点0'],
			[31, '随机角度旋转0_添加随机线条及点1'],
			[32, '随机角度旋转0_添加随机线条及点2'],
			[33, '随机角度旋转0_局部阈值二值化'],
			[34, '随机角度旋转0_提取特征值'],
			[35, '随机角度旋转0_随机放缩0'],
			[36, '随机角度旋转0_随机放缩1'],
			[37, '随机角度旋转0_随机放缩2'],
			[38, '随机角度旋转0_随机放缩3'],
			[39, '随机角度旋转0_随机放缩4'],
			[40, '随机角度旋转1_全局阈值二值化'],
			[41, '随机角度旋转1_高斯模糊1'],
			[42, '随机角度旋转1_高斯模糊2'],
			[43, '随机角度旋转1_随机颜色0'],
			[44, '随机角度旋转1_随机颜色1'],
			[45, '随机角度旋转1_随机颜色2'],
			[46, '随机角度旋转1_随机颜色3'],
			[47, '随机角度旋转1_随机颜色4'],
			[48, '随机角度旋转1_随机颜色5'],
			[49, '随机角度旋转1_随机颜色6'],
			[50, '随机角度旋转1_随机颜色7'],
			[51, '随机角度旋转1_灰度化'],
			[52, '随机角度旋转1_添加随机线条及点0'],
			[53, '随机角度旋转1_添加随机线条及点1'],
			[54, '随机角度旋转1_添加随机线条及点2'],
			[55, '随机角度旋转1_局部阈值二值化'],
			[56, '随机角度旋转1_提取特征值'],
			[57, '随机角度旋转1_随机放缩0'],
			[58, '随机角度旋转1_随机放缩1'],
			[59, '随机角度旋转1_随机放缩2'],
			[60, '随机角度旋转1_随机放缩3'],
			[61, '随机角度旋转1_随机放缩4'],
			[62, '随机角度旋转2_全局阈值二值化'],
			[63, '随机角度旋转2_高斯模糊1'],
			[64, '随机角度旋转2_高斯模糊2'],
			[65, '随机角度旋转2_随机颜色0'],
			[66, '随机角度旋转2_随机颜色1'],
			[67, '随机角度旋转2_随机颜色2'],
			[68, '随机角度旋转2_随机颜色3'],
			[69, '随机角度旋转2_随机颜色4'],
			[70, '随机角度旋转2_随机颜色5'],
			[71, '随机角度旋转2_随机颜色6'],
			[72, '随机角度旋转2_随机颜色7'],
			[73, '随机角度旋转2_灰度化'],
			[74, '随机角度旋转2_添加随机线条及点0'],
			[75, '随机角度旋转2_添加随机线条及点1'],
			[76, '随机角度旋转2_添加随机线条及点2'],
			[77, '随机角度旋转2_局部阈值二值化'],
			[78, '随机角度旋转2_提取特征值'],
			[79, '随机角度旋转2_随机放缩0'],
			[80, '随机角度旋转2_随机放缩1'],
			[81, '随机角度旋转2_随机放缩2'],
			[82, '随机角度旋转2_随机放缩3'],
			[83, '随机角度旋转2_随机放缩4'],
			[84, '随机放缩0'],
			[85, '随机放缩1'],
			[86, '随机放缩2'],
			[87, '随机放缩3'],
			[88, '随机放缩4']]
//翻页
function getPage(pn){
    var pageCount= Math.ceil(dataCount / pageSize);//总页数
    if(pn==0||pn>pageCount){
        return;
    }
    var ul=$(".listul");
    ul.empty();
    //console.log(pageCount+"..."+pn)
    paintPage(pageCount,pn);   //绘制页码
    var startPage = pageSize * (pn - 1);
    if (pageCount == 1) {     // 当只有一页时
        for (var j = 0; j < dataCount; j++) {
            var x = "../static/upload/"+timeStamp+"/augment/"+timeStamp+"/img@" + (j+1) + ".png";
            var e="<li><a href=\"content.html\"><img src=\""+ x +"\" width=\"160px\" height=\"145px\"></a><p><span>"+array[j][1]+"</span></p></li>";
            ul.append(e);
        }
    }else {      // 当超过一页时
        var e="";
        var endPage = pn<pageCount?pageSize * pn:dataCount;
        for (var j = startPage; j < endPage; j++) {
        	var x = "../static/upload/"+timeStamp+"/augment/"+timeStamp+"/img@" + (j+1) + ".png";
            var id = "img(" + (j+1) + ")"
            //var e="<li><a href=\"content.html\"><img src=\""+ x +"\"></a><p><span>图片"+j+"</span></p></li>";
            var e="<li><a href=\"content.html\"><img src=\""+ x +"\" width=\"160px\" height=\"145px\"></a><p><span>"+array[j][1]+"</span></p></li>";
            ul.append(e);
        }
    }
}

//绘制页码
function paintPage(number,currNum)  //number 总页数,currNum 当前页
{
    var pageUl=$(".fenye");
    pageUl.empty();
    var ulDetail="";

    if(number==1){
        ulDetail= "<li class=\"prev\"><a href=\"javascript:getPage(1)\">首页</a></li>" +
            "<li class=\"prev\"><a href=\"javascript:void(0)\">上一页</a></li>"+
            "<li class=\"numb choose\"><a href=\"javascript:getPage(1)\">1</a></li>"+
            "<li class=\"next\"><a href=\"javascript:void(0)\">下一页</a></li>"+
            "<li class=\"prev\"><a href=\"javascript:getPage(1)\">末页</a></li>";
    }else if(number==2){
        ulDetail= "<li class=\"prev\"><a href=\"javascript:getPage(1)\">首页</a></li>" +
            "<li class=\"prev\"><a href=\"javascript:getPage(1)\">上一页</a></li>"+
            "<li class=\"numb"+choosele(currNum,1)+"\"><a href=\"javascript:getPage(1)\">1</a></li>"+
            "<li class=\"numb"+choosele(currNum,2)+"\"><a href=\"javascript:getPage(2)\">2</a></li>"+
            "<li class=\"next\"><a href=\"javascript:getPage(2)\">下一页</a></li>"+
            "<li class=\"prev\"><a href=\"javascript:getPage(2)\">末页</a></li>";
    }else if(number==3){
        ulDetail= "<li class=\"prev\"><a href=\"javascript:getPage(1)\">首页</a></li>" +
            "<li class=\"prev\"><a href=\"javascript:getPage("+parseInt(currNum-1)+")\">上一页</a></li>"+
            "<li class=\"numb"+choosele(currNum,1)+"\"><a href=\"javascript:getPage(1)\">1</a></li>"+
            "<li class=\"numb"+choosele(currNum,2)+"\"><a href=\"javascript:getPage(2)\">2</a></li>"+
            "<li class=\"numb"+choosele(currNum,3)+"\"><a href=\"javascript:getPage(3)\">3</a></li>"+
            "<li class=\"next\"><a href=\"javascript:getPage("+parseInt(currNum+1)+")\">下一页</a></li>"+
            "<li class=\"prev\"><a href=\"javascript:getPage(3)\">末页</a></li>";
    }else if(number==currNum&&currNum>3){
        ulDetail= "<li class=\"prev\"><a href=\"javascript:getPage(1)\">首页</a></li>" +
            "<li class=\"prev\"><a href=\"javascript:getPage("+parseInt(currNum-1)+")\">上一页</a></li>"+
            "<li class=\"numb\"><a href=\"javascript:getPage("+parseInt(currNum-2)+")\">"+parseInt(currNum-2)+"</a></li>"+
            "<li class=\"numb\"><a href=\"javascript:getPage("+parseInt(currNum-1)+")\">"+parseInt(currNum-1)+"</a></li>"+
            "<li class=\"numb choose\"><a href=\"javascript:getPage("+currNum+")\">"+currNum+"</a></li>"+
            "<li class=\"next\"><a href=\"javascript:getPage("+currNum+")\">下一页</a></li>"+
            "<li class=\"prev\"><a href=\"javascript:getPage("+number+")\">末页</a></li>";
    }else if(currNum==1&&number>3){
        ulDetail= "<li class=\"prev\"><a href=\"javascript:getPage(1)\">首页</a></li>" +
            "<li class=\"prev\"><a href=\"javascript:void(0)\">上一页</a></li>"+
            "<li class=\"numb choose\"><a href=\"javascript:void(0)\">1</a></li>"+
            "<li class=\"numb\"><a href=\"javascript:getPage(2)\">2</a></li>"+
            "<li class=\"numb\"><a href=\"javascript:getPage(3)\">3</a></li>"+
            "<li class=\"next\"><a href=\"javascript:getPage(2)\">下一页</a></li>"+
            "<li class=\"prev\"><a href=\"javascript:getPage("+number+")\">末页</a></li>";
    }else{
        ulDetail= "<li class=\"prev\"><a href=\"javascript:getPage(1)\">首页</a></li>" +
            "<li class=\"prev\"><a href=\"javascript:getPage("+parseInt(currNum-1)+")\">上一页</a></li>"+
            "<li class=\"numb\"><a href=\"javascript:getPage("+parseInt(currNum-1)+")\">"+parseInt(currNum-1)+"</a></li>"+
            "<li class=\"numb choose\"><a href=\"javascript:getPage("+currNum+")\">"+currNum+"</a></li>"+
            "<li class=\"numb\"><a href=\"javascript:getPage("+parseInt(currNum+1)+")\">"+parseInt(currNum+1)+"</a></li>"+
            "<li class=\"next\"><a href=\"javascript:getPage("+parseInt(currNum+1)+")\">下一页</a></li>"+
            "<li class=\"prev\"><a href=\"javascript:getPage("+number+")\">末页</a></li>"
    }

    $(".fenye").append(ulDetail);

}

function choosele(num,cur){
    if(num==cur){
        return " choose";
    }else{
        return "";
    }
}

// 给img图片标签添加onClick=“bigImg（this）”事件
function bigImg(obj) {
    var image = new Image(); //创建一个image对象
    var path = obj.src;
    image.src=path;   //新创建的image添加src
    var width = image.width;  // 获取原始图片的宽
    var hight = image.height; // 获取原始图片高
    $("#bigImg").attr('src',path);
    $(".show-bigImg").css({"margin-top":-hight/2,"margin-left":-width/2});// 居中；css中使用了fixed定位top：50%；left：50%；
    $(".mengceng").css("display","block");
    if (width>1200) {
        $(".show-bigImg").css({"margin-left":-1200/2});
    }
    if (hight>800) {
        $(".show-bigImg").css({"margin-top":-800/2});
    }
}
// 点击放大图片关闭
function closeImg(obj) {
    $("#bigImg").attr('src',"");
    $(".mengceng").css("display","none");
}


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
	var formData = new FormData(); 
	//alert(pic);
	formData.append("file", pic);  //添加图片信息的参数
	$.ajax({
	    url: 'imageAugment',
	    type: 'POST',
	    async : true,//是否异步请求
	    cache: false, //上传文件不需要缓存
	    data: formData,
	    processData: false, // 告诉jQuery不要去处理发送的数据
	    contentType: false, // 告诉jQuery不要去设置Content-Type请求头
	    success: function (data) {
	    	if(data=="error")
	    		alert("扩展失败");
	    	else{
	    		var arr = data.split("|");
	    		dataCount = parseInt(arr[0]);
	    		timeStamp = arr[1];
	    		alert("扩展完成");
	    		getPage(1);
	    	} 		
	    },
	    error: function (data) {
	        alert("上传失败");
	    }
	})  
}