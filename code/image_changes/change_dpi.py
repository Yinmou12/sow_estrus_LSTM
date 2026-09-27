from PIL import Image


def check_and_update_dpi(input_filename, output_filename, target_dpi=1000):
    try:
        # 1. 打开原始图片
        img = Image.open(input_filename)

        # 2. 获取当前 DPI（如果图片元数据中没有 DPI 信息，则默认为 None）
        current_dpi = img.info.get("dpi")

        if current_dpi:
            print(f"图片 '{input_filename}' 的当前 DPI 为: {current_dpi}")
        else:
            print(
                f"图片 '{input_filename}' 未包含 DPI 元数据信息（通常默认为 72 或 96）。"
            )

        # 3. 如果当前 DPI 不是目标 DPI，则生成新图片
        if current_dpi != (target_dpi, target_dpi):
            # 另存为新文件，并强制写入 target_dpi DPI
            img.save(output_filename, dpi=(target_dpi, target_dpi))
            print(
                f"\n已成功生成新图片: '{output_filename}'，其 DPI 已设置为 {target_dpi}。"
            )
        else:
            print(f"\n该图片的 DPI 已经是 {target_dpi}，无需重新生成。")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{input_filename}'。请确保它与此脚本在同一目录下。")
    except Exception as e:
        print(f"处理图片时发生错误: {e}")
