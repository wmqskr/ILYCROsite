# Remove fake code X
def remove_O_from_file(filename):
    # Read the original file and remove all 'X' characters
    with open(filename, 'r') as file:
        content = file.read()
        content_without_O = content.replace('O', '')
        content_without_U = content_without_O.replace('U', '')
    # Write the processed data to a temporary file
    with open('./dataset/test/Kcr_INDP_cleaned.txt', 'w') as cleaned_file:
        cleaned_file.write(content_without_U)

# Call the function
# remove_O_from_file('./dataset/train/kcr_cvN.txt')
# remove_O_from_file('./dataset/train/kcr_cvP.txt')
# remove_O_from_file('./dataset/test/Kcr_INDN.txt')
remove_O_from_file('./dataset/test/Kcr_INDP.txt')