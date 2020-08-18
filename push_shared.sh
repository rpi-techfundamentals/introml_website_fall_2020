cp /Users/jasonkuruzovich/Box/Documents/Into-ml-app/book_fall_2020.xlsx   ./introml/book.xlsx
source build.sh
git add --all
if [ -z "$1"]
then
	git commit -m "push to public"
else
	git commit -m "$1"
fi
git push
