#include <lua.hpp>
#include <opencv2/opencv.hpp>
#include <Windows.h>
#include <iostream>

struct Pixel_RGBA{
	unsigned char b = 0;
	unsigned char g = 0;
	unsigned char r = 0;
	unsigned char a = 0;
};

namespace noise {
	inline double randxxh32(unsigned long input, unsigned long seed) {
		const unsigned long prime32_2 = 2246822519;
		const unsigned long prime32_3 = 3266489917;
		const unsigned long prime32_4 = 668265263;
		const unsigned long prime32_5 = 374761393;
		unsigned long h32 = input * prime32_3;
		h32 += seed + prime32_5 + 4;
		h32 = (h32 << 17) | (h32 >> 15);
		h32 *= prime32_4;
		h32 ^= h32 >> 15;
		h32 *= prime32_2;
		h32 ^= h32 >> 13;
		h32 *= prime32_3;
		h32 ^= h32 >> 16;
		return static_cast<double>(h32);
	}

	// randxxh32を指定の範囲内に収める乱数生成関数
	inline double rand(double minv, double maxv, unsigned long input, unsigned long seed) {
		double norm = randxxh32(input, seed) / ULONG_MAX;
		double range = maxv - minv;
		return norm * range + minv;
	}
}

int deterioration(lua_State *L){
	Pixel_RGBA *src = (Pixel_RGBA*)lua_touserdata(L, 1);
	int w = lua_tointeger(L, 2);
	int h = lua_tointeger(L, 3);
	int quality = lua_tointeger(L, 4);
	
	cv::Mat msrc(h, w, CV_8UC4, src);
	std::vector<int> param = { CV_IMWRITE_JPEG_QUALITY, quality };
	std::vector<unsigned char> binary;

	cv::imencode(".jpg", msrc, binary, param);
	cv::Mat dst = cv::imdecode(cv::Mat(binary), CV_LOAD_IMAGE_COLOR);

	int y;
#pragma omp parallel for schedule(guided)
	for (y = 0; y < h; ++y){
		cv::Vec3b *dstptr = dst.ptr<cv::Vec3b>(y);
		for (int x = 0; x < w; ++x){
			unsigned long pos = y*w + x;
			src[pos].b = dstptr[x][0];
			src[pos].g = dstptr[x][1];
			src[pos].r = dstptr[x][2];
		}
	}
	return 0;
}

int DQTGlitch(lua_State *L){
	Pixel_RGBA *src = (Pixel_RGBA*)lua_touserdata(L, 1);
	int w = lua_tointeger(L, 2);
	int h = lua_tointeger(L, 3);
	unsigned long seed = lua_tointeger(L, 4);
	int quality = lua_tointeger(L, 5);
	int N = lua_tointeger(L, 6);

	/* Matを作成 */
	cv::Mat msrc(h, w, CV_8UC4, src);
	std::vector<uchar> binary;
	/* Matをjpegでエンコード */
	cv::imencode(".jpg", msrc, binary, { CV_IMWRITE_JPEG_QUALITY, quality });
	/* バイナリを書き換える */
//#pragma omp parallel for schedule(guided)
	for (int i = 0; i < N; ++i){
		binary[static_cast<uint>(noise::rand(25, 88, seed, i))] = static_cast<uchar>(noise::rand(0, 255, seed << 1, i));
		binary[static_cast<uint>(noise::rand(94, 157, seed << 2, i))] = static_cast<uchar>(noise::rand(0, 255, seed << 3, i));
	}

	/* デコード */
	cv::Mat dst = cv::imdecode(cv::Mat(binary), CV_LOAD_IMAGE_COLOR);

	/* 値流し込み */
	int y;
#pragma omp parallel for schedule(guided)
	for (y = 0; y < h; ++y){
		cv::Vec3b *dstptr = dst.ptr<cv::Vec3b>(y);
		for (int x = 0; x < w; ++x){
			unsigned long pos = y*w + x;
			src[pos].b = dstptr[x][0];
			src[pos].g = dstptr[x][1];
			src[pos].r = dstptr[x][2];
		}
	}
	return 0;
}

int DHTGlitch(lua_State *L){
	Pixel_RGBA* src = (Pixel_RGBA*)lua_touserdata(L, 1);
	int w = lua_tointeger(L, 2); int h = lua_tointeger(L, 3);
	unsigned long seed = lua_tointeger(L, 4); int quality = lua_tointeger(L, 5);
	int N = lua_tointeger(L, 6);

	/* Matを作成 */
	cv::Mat msrc(h, w, CV_8UC4, src);
	std::vector<uchar> binary;
	/* Matをjpegでエンコード */
	cv::imencode(".jpg", msrc, binary, { CV_IMWRITE_JPEG_QUALITY, quality });
	/* バイナリを書き換える */
//#pragma omp parallel for schedule(guided)
	for (int i = 0; i < N; ++i){
		binary[static_cast<uint>(noise::rand(231, 391, seed, i))] = static_cast<uchar>(noise::rand(0, 255, seed << 1, i));
		binary[static_cast<uint>(noise::rand(447, 607, seed << 2, i))] = static_cast<uchar>(noise::rand(0, 255, seed << 3, i));
	}
	/* デコード */
	cv::Mat dst = cv::imdecode(cv::Mat(binary), CV_LOAD_IMAGE_COLOR);

	/* 値流し込み */
	int x, y;
#pragma omp parallel for private(x) schedule(guided)
	for (y = 0; y < h; ++y){
		cv::Vec3b *dstptr = dst.ptr<cv::Vec3b>(y);
		for (x = 0; x < w; ++x){
			unsigned long pos = y*w + x;
			src[pos].b = dstptr[x][0];
			src[pos].g = dstptr[x][1];
			src[pos].r = dstptr[x][2];
		}
	}
	return 0;
}

int ImageGlitch(lua_State *L){
	Pixel_RGBA* src = (Pixel_RGBA*)lua_touserdata(L, 1);
	int w = lua_tointeger(L, 2); int h = lua_tointeger(L, 3);
	unsigned long seed = lua_tointeger(L, 4); int quality = lua_tointeger(L, 5);
	int N = lua_tointeger(L, 6);

	/* Matを作成 */
	cv::Mat msrc(h, w, CV_8UC4, src);
	std::vector<uchar> binary;
	/* Matをjpegでエンコード */
	cv::imencode(".jpg", msrc, binary, { CV_IMWRITE_JPEG_QUALITY, quality });

	/* バイナリを書き換える */
//#pragma omp parallel for schedule(guided)
	for (int i = 0; i < N; ++i){
		binary[static_cast<uint>(noise::rand(623, binary.size() - 8, seed, i))] = static_cast<uchar>(noise::rand(0, 255, seed << 1, i));
	}

	/* デコード */
	cv::Mat dst = cv::imdecode(cv::Mat(binary), CV_LOAD_IMAGE_COLOR);

	/* 値流し込み */
	int x, y;
#pragma omp parallel for private(x) schedule(guided)
	for (y = 0; y < h; ++y){
		cv::Vec3b *dstptr = dst.ptr<cv::Vec3b>(y);
		for (x = 0; x < w; ++x){
			unsigned long pos = y*w + x;
			src[pos].b = dstptr[x][0];
			src[pos].g = dstptr[x][1];
			src[pos].r = dstptr[x][2];
		}
	}
	return 0;
}

static luaL_Reg FuncList[] = {
	{ "deterioration", deterioration },
	{ "DQTGlitch", DQTGlitch },
	{ "DHTGlitch", DHTGlitch },
	{ "ImageGlitch",ImageGlitch },
	{ NULL, NULL }
};

extern "C"{
	__declspec(dllexport) int luaopen_JpgGlitch(lua_State* L) {
		luaL_register(L, "JpgGlitch", FuncList);
		return 1;
	}
}